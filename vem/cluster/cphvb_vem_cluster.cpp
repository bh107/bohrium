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

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <cphvb.h>

#include "cphvb_vem_cluster.h"
#include "exchange.h"
#include "mapping.h"
#include "darray_extension.h"

//Function pointers to the Node VEM.
static cphvb_init vem_init;
static cphvb_execute vem_execute;
static cphvb_shutdown vem_shutdown;
static cphvb_reg_func vem_reg_func;

//The VE components
static cphvb_component **my_components;

//Our self
static cphvb_component *myself;

//Number of user-defined functions registered.
static cphvb_intp userfunc_count = 0;

/* Initialize the VEM
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error cphvb_vem_cluster_init(cphvb_component *self)
{
    cphvb_intp children_count;
    cphvb_error err;
    myself = self;

    err = cphvb_component_children(self, &children_count, &my_components);
    if (children_count != 1) 
    {
		std::cerr << "Unexpected number of child nodes for VEM, must be 1" << std::endl;
		return CPHVB_ERROR;
    }
    
    if (err != CPHVB_SUCCESS)
	    return err;
    
    vem_init = my_components[0]->init;
    vem_execute = my_components[0]->execute;
    vem_shutdown = my_components[0]->shutdown;
    vem_reg_func = my_components[0]->reg_func;

    //Let us initiate the Node VEM.
    if((err = vem_init(my_components[0])) != 0)
        return err;

    return CPHVB_SUCCESS;
}


/* Shutdown the VEM, which include a instruction flush
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error cphvb_vem_cluster_shutdown(void)
{
    cphvb_error err;
    err = vem_shutdown();
    cphvb_component_free(my_components[0]);//Only got one child.
    vem_init     = NULL;
    vem_execute  = NULL;
    vem_shutdown = NULL;
    vem_reg_func = NULL;
    cphvb_component_free_ptr(my_components);
    my_components = NULL;
    return err;
}


/* Register a new user-defined function.
 *
 * @lib Name of the shared library e.g. libmyfunc.so
 *      When NULL the default library is used.
 * @fun Name of the function e.g. myfunc
 * @id Identifier for the new function. The bridge should set the
 *     initial value to Zero. (in/out-put)
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error cphvb_vem_cluster_reg_func(char *fun, cphvb_intp *id)
{
    if(*id == 0)//Only if parent didn't set the ID.
        *id = userfunc_count + 1;

    return vem_reg_func(fun, id);
}


/* Execute a list of instructions (blocking, for the time being).
 * It is required that the VEM supports all instructions in the list.
 *
 * @instruction A list of instructions to execute
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error cphvb_vem_cluster_execute(cphvb_intp count,
                                      cphvb_instruction inst_list[])
{
    //Local copy of the instruction list
    std::vector<cphvb_instruction> local_inst;
    local_inst.assign(inst_list, inst_list + count);
    int NPROC = 1;
    
    if (count <= 0)
        return CPHVB_SUCCESS;
    
    #ifdef CPHVB_TRACE
        for(cphvb_intp i=0; i<count; ++i)
        {
            cphvb_instruction* inst = &inst_list[i];
            cphvb_component_trace_inst(myself, inst);
        }
    #endif
    
    //Exchange the instruction list between all processes and
    //update all operand pointers to local pointers
    exchange_inst_list(count, &local_inst.front());


    cphvb_pprint_instr_list(&local_inst.front(), count, "CLUSTER");


    for(cphvb_intp i=0; i < count; ++i)
    {
        cphvb_instruction* inst = &local_inst[i];
        assert(inst->opcode >= 0);
        switch(inst->opcode) 
        {
            case CPHVB_DISCARD:
            case CPHVB_FREE:
            case CPHVB_SYNC:
            case CPHVB_NONE:
            case CPHVB_USERFUNC:
            {
                local_inst[i].status = vem_execute(1, inst);
                if(local_inst[i].status != CPHVB_SUCCESS)
                    return local_inst[i].status;
                break;
            }
            default:
            if(0)
            {   
                std::vector<cphvb_array> chunks;
                std::vector<darray_ext> chunks_ext;
                local_array(NPROC, inst->operand[0], &chunks, &chunks_ext);
                int nchunks_per_operand = (int) chunks.size();
                for(cphvb_intp j=1; j < cphvb_operands_in_instruction(inst); ++j)
                {
                    local_array(NPROC, inst->operand[j], &chunks, &chunks_ext);
                }
                
                for(cphvb_intp j=0; j < nchunks_per_operand; ++j)
                {
                    cphvb_intp size=1;
                    for(cphvb_intp k=0; k<cphvb_base_array(inst->operand[0])->ndim; ++k)
                        size *= cphvb_base_array(inst->operand[0])->shape[k];
                    size = size / NPROC;

                    cphvb_instruction new_inst = *inst;
                    for(cphvb_intp k=0; k < cphvb_operands_in_instruction(inst); ++k)
                    {
                        if(!cphvb_is_constant(inst->operand[k]))
                        {
                            int c = j+k*nchunks_per_operand;
                            chunks[c].start += size * chunks_ext[c].rank; 
                            new_inst.operand[k] = &chunks[c];
                        }
                    }
                    cphvb_error e = vem_execute(1, &new_inst);
                    if(e != CPHVB_SUCCESS)
                    {
                        return e;
                    }
                }
                break;
            }
            {
                std::vector<cphvb_array> chunks;
                std::vector<darray_ext> chunks_ext;
                cphvb_error e = local_arrays(NPROC, inst, chunks, chunks_ext);
                if(e != CPHVB_SUCCESS)
                    return e;
                
                
                

                break; 
            }
        }
    }

    return CPHVB_SUCCESS;
}
