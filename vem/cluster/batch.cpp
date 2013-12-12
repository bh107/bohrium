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
#include <set>
#include "task.h"
#include "exec.h"
#include "except.h"
#include "pgrid.h"
#include "comm.h"
#include "timing.h"

static std::vector<task> task_store;
static std::set<bh_base *> discard_store;

/* Schedule an task.
 * @t  The task to schedule
 */
void batch_schedule(const task& t)
{
    task_store.push_back(t);
}


/* Schedule an instruction
 * @inst   The instruction to schedule
 */
void batch_schedule_inst(const bh_instruction& inst)
{
    task t;
    t.inst.type = TASK_INST;
    t.inst.inst = inst;
    batch_schedule(t);
}


/* Schedule an instruction that only takes one operand.
 *
 * @opcode   The opcode of the instruction
 * @base  The local base array in the instruction
 */
void batch_schedule_inst_on_base(bh_opcode opcode, bh_base *base)
{
    //insert returns True if the operand didn't exist in the discard_store
    if(opcode == BH_DISCARD && !discard_store.insert(base).second)
        return;//Avoid discarding a base array multiple times

    task t;
    t.inst.type = TASK_INST;
    t.inst.inst.opcode = opcode;
    bh_assign_complete_base(&t.inst.inst.operand[0], base);
    assert(bh_operands_in_instruction(&t.inst.inst) == 1);
    batch_schedule(t);
}


/* Schedule an instruction.
 *
 * @opcode   The opcode of the instruction
 * @operands The local operands in the instruction
 */
void batch_schedule_inst(bh_opcode opcode, bh_view *operands)
{
    task t;
    t.inst.type = TASK_INST;
    t.inst.inst.opcode = opcode;
    memcpy(t.inst.inst.operand, operands, bh_operands(opcode)
                                          * sizeof(bh_view));
    batch_schedule(t);
}


/* Schedule an send/receive instruction.
 *
 * @direction   If True the array is send else it is received.
 * @rank        The process to send to or receive from
 * @local_view  The local view to communicate
 *              NB: it must be contiguous (row-major)
 */
void batch_schedule_comm(bool direction, int rank, const bh_view &local_view)
{
    task t;
    t.send_recv.type       = TASK_SEND_RECV;
    t.send_recv.direction  = direction;
    t.send_recv.rank       = rank;
    t.send_recv.local_view = local_view;
    batch_schedule(t);
}


/* Flush all scheduled instructions
 *
 */
void batch_flush()
{
    bh_uint64 stime = bh_timing();

    for(std::vector<task>::iterator it=task_store.begin();
        it != task_store.end(); ++it)
    {
        switch((*it).inst.type)
        {
            case TASK_INST:
            {
                bh_error e;
                bh_instruction *inst = &((*it).inst.inst);
                bh_ir bhir;
                if((e = bh_ir_create(&bhir, 1, inst)) != BH_SUCCESS)
                    EXCEPT("bh_ir_create() failed");

                if((e = mychild->execute(&bhir)) != BH_SUCCESS)
                    EXCEPT_INST(inst->opcode, e);
                bh_ir_destroy(&bhir);

                if(inst->opcode == BH_DISCARD)
                    array_rm_local(bh_base_array(&inst->operand[0]));
                break;
            }
            case TASK_SEND_RECV:
            {
                bh_error e;
                bool dir      = (*it).send_recv.direction;
                int rank      = (*it).send_recv.rank;
                bh_view *view = &(*it).send_recv.local_view;
                bh_intp s     = bh_type_size(bh_base_array(view)->type);
                bh_intp nelem = bh_nelements_nbcast(view);

                bh_uint64 stime = bh_timing();
                if(dir)//Sending
                {
                    char *data = (char*) view->base->data;
                    assert(data != NULL);
                    data += view->start * s;
                    if((e = MPI_Send(data, nelem * s, MPI_BYTE, rank, 0,
                                     MPI_COMM_WORLD)) != MPI_SUCCESS)
                    EXCEPT_MPI(e);
                }
                else//Receiving
                {
                    if((e = bh_data_malloc(view->base)) != BH_SUCCESS)
                        EXCEPT_OUT_OF_MEMORY();
                    char *data = (char*) view->base->data;
                    data += view->start * s;
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
    discard_store.clear();
    bh_timing_save(timing_flush, stime, bh_timing());
}




