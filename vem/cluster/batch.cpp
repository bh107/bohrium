/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with Bohrium. 

If not, see <http://www.gnu.org/licenses/>.
*/

#include <cassert>
#include <StaticStore.hpp>
#include <bh.h>
#include <vector>
#include "task.h"
#include "exec.h"
#include "except.h"
#include "pgrid.h"
#include "comm.h"
#include "timing.h"

static std::vector<task> task_store;


/* Schedule an task. 
 * NB: for now, we will flush in every task scheduling
 * @t  The task to schedule 
 */
void batch_schedule(const task& t)
{
    task_store.push_back(t);
}


/* Schedule an instruction
 * @inst   The instruction to schedule 
 */
void batch_schedule(const bh_instruction& inst)
{
    task t;
    t.inst.type = TASK_INST;
    t.inst.inst = inst;
    batch_schedule(t); 
}


/* Schedule an instruction that only takes one instruction.
 *
 * @opcode   The opcode of the instruction
 * @operand  The local operand in the instruction
 */
void batch_schedule(bh_opcode opcode, bh_array *operand)
{
    task t;
    t.inst.type = TASK_INST;
    t.inst.inst.opcode = opcode;
    t.inst.inst.operand[0] = operand;
    assert(bh_operands_in_instruction(&t.inst.inst) == 1);
    batch_schedule(t); 
}


/* Schedule an instruction.
 *
 * @opcode   The opcode of the instruction
 * @operands The local operands in the instruction
 * @ufunc    The user-defined function struct when opcode is BH_USERFUNC.
 */
void batch_schedule(bh_opcode opcode, bh_array *operands[],
                    bh_userfunc *ufunc)
{
    task t;
    t.inst.type = TASK_INST;
    t.inst.inst.opcode = opcode;
    t.inst.inst.userfunc = ufunc;
    if(ufunc == NULL)
    {
        assert(opcode != BH_USERFUNC);
        memcpy(t.inst.inst.operand, operands, bh_operands(opcode) 
                                              * sizeof(bh_array*));
    }
    batch_schedule(t); 
}


/* Schedule an send/receive instruction.
 *
 * @direction  If True the array is send else it is received.
 * @rank       The to send to or receive from
 * @local_ary  The local array to communicate
 */
void batch_schedule(bool direction, int rank, bh_array *local_ary)
{
    task t;
    t.send_recv.type      = TASK_SEND_RECV;
    t.send_recv.direction = direction;
    t.send_recv.rank      = rank;
    t.send_recv.local_ary = local_ary;
    batch_schedule(t); 
}


/* Flush all scheduled instructions
 * 
 */
void batch_flush()
{
    for(std::vector<task>::iterator it=task_store.begin(); 
        it != task_store.end(); ++it)
    {
        switch((*it).inst.type) 
        {
            case TASK_INST:
            {
                bh_error e;
                bh_instruction *inst = &((*it).inst.inst);
                if((e = exec_vem_execute(1, inst)) != BH_SUCCESS)
                    EXCEPT_INST(inst->opcode, e);

                if(inst->opcode == BH_DISCARD)
                {
                    if(inst->operand[0]->base == NULL)
                        array_rm_local(inst->operand[0]); 
                }
                break;
            }
            case TASK_SEND_RECV:
            {
                bh_error e;
                bool dir         = (*it).send_recv.direction;
                int rank         = (*it).send_recv.rank;
                bh_array *ary = (*it).send_recv.local_ary;
                bh_intp s     = bh_type_size(ary->type);
                bh_intp nelem = bh_nelements(ary->ndim, ary->shape);
               
                bh_uint64 stime = bh_timing();
                if(dir)//Sending
                {
                    char *data = (char*) bh_base_array(ary)->data;
                    assert(data != NULL);
                    data += ary->start * s;
                    if((e = MPI_Send(data, nelem * s, MPI_BYTE, rank, 0, 
                                     MPI_COMM_WORLD)) != MPI_SUCCESS)
                    EXCEPT_MPI(e);
                }
                else//Receiving
                {
                    if((e = bh_data_malloc(ary)) != BH_SUCCESS)
                        EXCEPT_OUT_OF_MEMORY();
                    char *data = (char*) bh_base_array(ary)->data;
                    data += ary->start * s;
                    if((e = MPI_Recv(data, nelem * s, MPI_BYTE, rank, 
                                     0, MPI_COMM_WORLD, 
                                     MPI_STATUS_IGNORE)) != MPI_SUCCESS)
                        EXCEPT_MPI(e);
                }
                bh_timing_save(timing_comm_p2p, stime, bh_timing());
                break;
            }
            default:
            {
                fprintf(stderr, "[VEM-CLUSTER] batch_schedule encountered "
                        "an unknown task type\n");
                MPI_Abort(MPI_COMM_WORLD,BH_ERROR);
            }
        }        
    }
    task_store.clear();
}




