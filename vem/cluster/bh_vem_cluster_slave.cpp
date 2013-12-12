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

#include <bh.h>
#include <cassert>
#include <map>
#include "bh_vem_cluster.h"
#include "dispatch.h"
#include "pgrid.h"
#include "exec.h"
#include "timing.h"


//Check for error. Will exit on error.
static void check_error(bh_error err, const char *file, int line)
{
    if(err != BH_SUCCESS)
    {
        fprintf(stderr, "[VEM-CLUSTER] Slave (rank %d) encountered the error %s at %s:%d\n",
                pgrid_myrank, bh_error_text(err), file, line);
        MPI_Abort(MPI_COMM_WORLD,-1);
    }
}

int main()
{
    dispatch_msg *msg;

    timing_init();

    //Initiate the process grid
    pgrid_init();

    while(1)
    {
        //Receive the dispatch message from the master-process
        dispatch_reset();
        dispatch_recv(&msg);

        //Handle the message
        switch(msg->type)
        {
            case BH_CLUSTER_DISPATCH_INIT:
            {
                char *name = msg->payload;
                printf("Slave (rank %d) received INIT. name: %s\n", pgrid_myrank, name);
                check_error(exec_init(name),__FILE__,__LINE__);
                break;
            }
            case BH_CLUSTER_DISPATCH_SHUTDOWN:
            {
                printf("Slave (rank %d) received SHUTDOWN\n",pgrid_myrank);
                check_error(exec_shutdown(),__FILE__,__LINE__);
                return 0;
            }
            case BH_CLUSTER_DISPATCH_EXTMETHOD:
            {
                bh_opcode opcode = *((bh_opcode *)msg->payload);
                char *name = msg->payload+sizeof(bh_opcode);
                printf("Slave (rank %d) received UFUNC. fun: %s, id: %ld\n",pgrid_myrank, name, opcode);
                check_error(exec_extmethod(name, opcode),__FILE__,__LINE__);
                break;
            }
            case BH_CLUSTER_DISPATCH_EXEC:
            {
                //Deserialize the BhIRi
                bh_ir *bhir = (bh_ir*) msg->payload;
                bh_ir_deserialize(bhir);

                //The number of new arrays
                bh_intp *noa = (bh_intp *)(((char*)msg->payload)+bh_ir_totalsize(bhir));
                //The list of new arrays
                dispatch_array *darys = (dispatch_array*)(noa+1); //number of new arrays

                //Insert the new array into the array store and the array maps
                std::stack<bh_base*> base_darys;
                for(bh_intp i=0; i < *noa; ++i)
                {
                    bh_base *ary = dispatch_new_slave_array(&darys[i].ary, darys[i].id);
                    base_darys.push(ary);
                }

                //Receive the dispatched array-data from the master-process
                dispatch_array_data(base_darys);

                //Update all instruction to reference local arrays
                for(bh_intp i=0; i < bhir->ninstr; ++i)
                {
                    bh_instruction *inst = &bhir->instr_list[i];
                    int nop = bh_operands_in_instruction(inst);
                    bh_view *ops = bh_inst_operands(inst);

                    //Convert all instructon operands
                    for(bh_intp j=0; j<nop; ++j)
                    {
                        if(bh_is_constant(&ops[j]))
                            continue;
                        bh_base *base = bh_base_array(&ops[j]);
                        assert(dispatch_slave_exist((bh_intp)base));
                        bh_base_array(&ops[j]) = dispatch_master2slave((bh_intp)base);
                    }
                }
                check_error(exec_execute(bhir),__FILE__,__LINE__);
                break;
            }
            default:
                fprintf(stderr, "[VEM-CLUSTER] Slave (rank %d) "
                        "received unknown message type\n", pgrid_myrank);
                MPI_Abort(MPI_COMM_WORLD,BH_ERROR);
        }
    }
    return BH_SUCCESS;
}
