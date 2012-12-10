/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/

#include <cphvb.h>
#include <cassert>
#include <map>
#include "cphvb_vem_cluster.h"
#include "dispatch.h"
#include "pgrid.h"
#include "exec.h"


//Check for error. Will exit on error.
static void check_error(cphvb_error err, const char *file, int line)
{
    if(err != CPHVB_SUCCESS)
    {
        fprintf(stderr, "[VEM-CLUSTER] Slave (rank %d) encountered the error %s at %s:%d\n",
                pgrid_myrank, cphvb_error_text(err), file, line);
        MPI_Abort(MPI_COMM_WORLD,-1);
    }
}

//Check for execution error. Will exit on error.
static void check_exec_error(cphvb_error err, const char *file, int line, 
                             cphvb_intp count, cphvb_instruction inst_list[])
{
    if(err == CPHVB_PARTIAL_SUCCESS)//Only partial success
    {
        char msg[count+4096];
        sprintf(msg, "[VEM-CLUSTER] Slave (rank %d) encountered error in instruction batch: %s\n",
                pgrid_myrank, cphvb_error_text(err));
        for(cphvb_intp i=0; i<count; ++i)
        {
            cphvb_instruction *ist = &inst_list[i];
            sprintf(msg+strlen(msg),"\tOpcode: %s", cphvb_opcode_text(ist->opcode));
            if(ist->opcode == CPHVB_USERFUNC)
            {
                sprintf(msg+strlen(msg),", Operand types:");
                for(cphvb_intp j=0; j<cphvb_operands_in_instruction(ist); ++j)
                    sprintf(msg+strlen(msg)," %s", cphvb_type_text(cphvb_type_operand(ist,j))); 
            }
            else
            {
                sprintf(msg+strlen(msg),", Operand types:");
                for(cphvb_intp j=0; j<cphvb_operands_in_instruction(ist); ++j)
                    sprintf(msg+strlen(msg)," %s", cphvb_type_text(cphvb_type_operand(ist,j))); 
            }
            sprintf(msg+strlen(msg),", Status: %s\n", cphvb_error_text(ist->status));
        }
        fprintf(stderr,"%s", msg);
        MPI_Abort(MPI_COMM_WORLD,-1);
    }
    check_error(err, file, line);
}  

int main()
{
    dispatch_msg *msg;
    
    //Initiate the process grid
    check_error(pgrid_init(),__FILE__,__LINE__);

    while(1)
    {
        //Receive the dispatch message from the master-process
        check_error(dispatch_reset(),__FILE__,__LINE__);
        check_error(dispatch_recv(&msg),__FILE__,__LINE__);

        //Handle the message
        switch(msg->type) 
        {
            case CPHVB_CLUSTER_DISPATCH_INIT:
            {
                char *name = msg->payload;
                printf("Slave (rank %d) received INIT. name: %s\n", pgrid_myrank, name);
                check_error(exec_init(name),__FILE__,__LINE__);
                break;
            }
            case CPHVB_CLUSTER_DISPATCH_SHUTDOWN:
            {
                printf("Slave (rank %d) received SHUTDOWN\n",pgrid_myrank);
                check_error(exec_shutdown(),__FILE__,__LINE__);
                return 0;
            }
            case CPHVB_CLUSTER_DISPATCH_UFUNC:
            {
                cphvb_intp *id = (cphvb_intp *)msg->payload;
                char *fun = msg->payload+sizeof(cphvb_intp);
                printf("Slave (rank %d) received UFUNC. fun: %s, id: %ld\n",pgrid_myrank, fun, *id);
                check_error(exec_reg_func(fun, id),__FILE__,__LINE__);
                break;
            }
            case CPHVB_CLUSTER_DISPATCH_EXEC:
            {
                //The number of instructions
                cphvb_intp *noi = (cphvb_intp *)msg->payload;                 
                //The master-instruction list
                cphvb_instruction *master_list = (cphvb_instruction *)(noi+1);
                //The number of new arrays
                cphvb_intp *noa = (cphvb_intp *)(master_list + *noi);
                //The list of new arrays
                dispatch_array *darys = (dispatch_array*)(noa+1); //number of new arrays
                //The number of user-defined functions
                cphvb_intp *nou = (cphvb_intp *)(darys + *noa);
                //The list of user-defined functions
                cphvb_userfunc *ufunc = (cphvb_userfunc*)(nou+1); //number of new arrays


//   printf("Slave (rank %d) received EXEC. noi: %ld, noa: %ld, nou: %ld\n",pgrid_myrank, *noi, *noa, *nou);
               
                //Insert the new array into the array store and the array maps
                std::stack<cphvb_array*> base_darys;
                for(cphvb_intp i=0; i < *noa; ++i)
                {
                    cphvb_array *ary = dispatch_new_slave_array(&darys[i].ary, darys[i].id);
                    if(ary->base == NULL)//This is a base array.
                        base_darys.push(ary);
                } 
                //Update the base-array-pointers
                for(cphvb_intp i=0; i < *noa; ++i)
                {
                    cphvb_array *ary = dispatch_master2slave(darys[i].id);
                    if(ary->base != NULL)//This is NOT a base array
                    {
                        assert(dispatch_slave_exist(((cphvb_intp)ary->base)));
                        ary->base = dispatch_master2slave((cphvb_intp)ary->base);
                    }
                }

                //Receive the dispatched array-data from the master-process
                check_error(dispatch_array_data(base_darys),__FILE__,__LINE__);
                    
                    
                //Allocate the local instruction list that should reference local arrays
                cphvb_instruction *local_list = (cphvb_instruction *)malloc(*noi*sizeof(cphvb_instruction));
                if(local_list == NULL)
                    check_error(CPHVB_OUT_OF_MEMORY,__FILE__,__LINE__);
        
                memcpy(local_list, master_list, (*noi)*sizeof(cphvb_instruction));
            
                //De-serialize all user-defined function pointers.
                for(cphvb_intp i=0; i < *noi; ++i)
                {
                    cphvb_instruction *inst = &local_list[i];
                    if(inst->opcode == CPHVB_USERFUNC)
                    {   
                        inst->userfunc = (cphvb_userfunc*) malloc(ufunc->struct_size);
                        if(inst->userfunc == NULL)
                            check_error(CPHVB_OUT_OF_MEMORY,__FILE__,__LINE__);
                        //Save a local copy of the user-defined function
                        memcpy(inst->userfunc, ufunc, ufunc->struct_size);
                        //Iterate to the next user-defined function
                        ufunc = (cphvb_userfunc*)(((char*)ufunc) + ufunc->struct_size);
                    }
                }

                //Update all instruction to reference local arrays 
                for(cphvb_intp i=0; i < *noi; ++i)
                {
                    cphvb_instruction *inst = &local_list[i];
                    int nop = cphvb_operands_in_instruction(inst);
                    cphvb_array **ops;
                    if(inst->opcode == CPHVB_USERFUNC)
                        ops = inst->userfunc->operand;
                    else
                        ops = inst->operand;

                    //Convert all instructon operands
                    for(cphvb_intp j=0; j<nop; ++j)
                    { 
                        if(cphvb_is_constant(ops[j]))
                            continue;
                        assert(dispatch_slave_exist((cphvb_intp)ops[j]));
                        ops[j] = dispatch_master2slave((cphvb_intp)ops[j]);
                    }
                }

//                cphvb_pprint_instr_list(local_list, *noi, "SLAVE");

                check_exec_error(exec_execute(*noi, local_list),__FILE__,__LINE__, *noi, local_list);
                free(local_list);
                break;
            }
            default:
                fprintf(stderr, "[VEM-CLUSTER] Slave (rank %d) "
                        "received unknown message type\n", pgrid_myrank);
                MPI_Abort(MPI_COMM_WORLD,CPHVB_ERROR);
        }
    }
    return CPHVB_SUCCESS; 
}
