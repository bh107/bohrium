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
#include <cstring>
#include <iostream>
#include <vector>
#include <set>
#include <cphvb.h>

#include "cphvb_vem_cluster.h"
#include "mapping.h"
#include "array.h"
#include "pgrid.h"
#include "dispatch.h"
#include "comm.h"
#include "except.h"
#include "ufunc_reduce.h"
#include "batch.h"
#include "tmp.h"


//Function pointers to the Node VEM.
static cphvb_init vem_init;
static cphvb_shutdown vem_shutdown;
static cphvb_reg_func vem_reg_func;
//Public function pointer to the Node VEM
cphvb_execute exec_vem_execute;

//The VE components
static cphvb_component **my_components;

//Our self
static cphvb_component *myself;

//Number of user-defined functions registered.
static cphvb_intp userfunc_count = 0;
//User-defined function IDs.
static cphvb_userfunc_impl reduce_impl = NULL;
static cphvb_intp reduce_impl_id = 0;


/* Initialize the VEM
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error exec_init(const char *component_name)
{
    cphvb_intp children_count;
    cphvb_error err;
    myself = cphvb_component_setup(component_name);
    if(myself == NULL)
        return CPHVB_ERROR;
    
    err = cphvb_component_children(myself, &children_count, &my_components);
    if (children_count != 1) 
    {
		std::cerr << "Unexpected number of child nodes for VEM, must be 1" << std::endl;
		return CPHVB_ERROR;
    }
    
    if (err != CPHVB_SUCCESS)
	    return err;
    
    vem_init = my_components[0]->init;
    exec_vem_execute = my_components[0]->execute;
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
cphvb_error exec_shutdown(void)
{
    cphvb_error err;
    if((err = vem_shutdown()) != CPHVB_SUCCESS)
        return err;
    cphvb_component_free(my_components[0]);//Only got one child.
    vem_init     = NULL;
    exec_vem_execute  = NULL;
    vem_shutdown = NULL;
    vem_reg_func = NULL;
    cphvb_component_free_ptr(my_components);
    my_components = NULL;

    //Finalize the process grid
    pgrid_finalize();

    //Finalize the process grid
    dispatch_finalize();

    return CPHVB_SUCCESS;
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
cphvb_error exec_reg_func(char *fun, cphvb_intp *id)
{
    cphvb_error e;
    
    if(*id == 0)//Only if parent didn't set the ID.
    {
        *id = ++userfunc_count;
        assert(pgrid_myrank == 0);
    }

    if((e = vem_reg_func(fun, id)) != CPHVB_SUCCESS)
    {
        *id = 0;
        return e;
    }

    if(strcmp("cphvb_reduce", fun) == 0)
    {
        if(reduce_impl == NULL)
        {
            cphvb_component_get_func(myself, fun, &reduce_impl);
            if (reduce_impl == NULL)
                return CPHVB_USERFUNC_NOT_SUPPORTED;

            reduce_impl_id = *id;
            return CPHVB_SUCCESS;           
        }
    }

    return CPHVB_SUCCESS;
}


/* Execute to instruction locally at the master-process
 *
 * @instruction The instructionto execute
 */
static void fallback_exec(cphvb_instruction *inst)
{
    int nop = cphvb_operands_in_instruction(inst);
    std::set<cphvb_array*> arys2discard;
    
    batch_flush();

    //Gather all data at the master-process
    cphvb_array **oprands = cphvb_inst_operands(inst);
    for(cphvb_intp o=0; o < nop; ++o)
    {
        cphvb_array *op = oprands[o];
        if(cphvb_is_constant(op))
            continue;

        cphvb_array *base = cphvb_base_array(op);
        comm_slaves2master(base);
    }
    
    //Do global instruction
    if(pgrid_myrank == 0)
    {
        batch_schedule(*inst);
    }

    //Scatter all data back to all processes
    for(cphvb_intp o=0; o < nop; ++o)
    {
        cphvb_array *op = oprands[o];
        if(cphvb_is_constant(op))
            continue;
        cphvb_array *base = cphvb_base_array(op);
        
        //We have to make sure that the master-process has allocated memory
        //because the slaves cannot determine it.
        if(pgrid_myrank == 0)        
            cphvb_data_malloc(base);
        
        comm_master2slaves(base);

        //All local arrays should be discarded
        arys2discard.insert(op);
        arys2discard.insert(base);
    }
    //Discard all local views
    for(std::set<cphvb_array*>::iterator it=arys2discard.begin(); 
        it != arys2discard.end(); ++it)
    {
        if((*it)->base != NULL)
        {
            batch_schedule(CPHVB_DISCARD, *it);
        }
    }    
    //Free and discard all local base arrays
    for(std::set<cphvb_array*>::iterator it=arys2discard.begin(); 
        it != arys2discard.end(); ++it)
    {
        if((*it)->base == NULL)
        {
            batch_schedule(CPHVB_FREE, *it);
            batch_schedule(CPHVB_DISCARD, *it);
        }
    }    
}


/* Execute a regular computation instruction
 *
 * @instruction The regular computation instruction
 */
static void execute_regular(cphvb_instruction *inst)
{
    std::vector<ary_chunk> chunks;
    int nop = cphvb_operands_in_instruction(inst);
    cphvb_array **operands = cphvb_inst_operands(inst);

    mapping_chunks(nop, operands, chunks);
    assert(chunks.size() > 0);
    
    //Handle one chunk at a time.
    for(std::vector<ary_chunk>::size_type c=0; c < chunks.size();c += nop)
    {
        assert(cphvb_nelements(chunks[0].ary->ndim, chunks[0].ary->shape) > 0);

        //The process where the output chunk is located will do the computation.
        int owner_rank = chunks[0+c].rank;

        //Create a local instruction based on the array-chunks
        cphvb_instruction local_inst = *inst;
        for(cphvb_intp k=0; k < nop; ++k)
        {
            if(!cphvb_is_constant(inst->operand[k]))
            {
                ary_chunk *chunk = &chunks[k+c];
                local_inst.operand[k] = chunk->ary;
                comm_array_data(chunk, owner_rank);
            }
        }

        //Check if we should do the computation
        if(pgrid_myrank != owner_rank)
            continue;

        //Apply the local computation
        local_inst.status = CPHVB_INST_PENDING;

        //Schedule task
        batch_schedule(local_inst);
    
        //Free and discard all local chunk arrays
        for(cphvb_intp k=0; k < nop; ++k)
        {
            if(cphvb_is_constant(inst->operand[k]))
                continue;
            
            cphvb_array *ary = chunks[k+c].ary;
            if(ary->base == NULL)
                batch_schedule(CPHVB_FREE, ary);
            batch_schedule(CPHVB_DISCARD, ary);
        }
    }
}



/* Execute a list of instructions where all operands are global arrays
 *
 * @instruction A list of instructions to execute
 * @return Error codes
 */
cphvb_error exec_execute(cphvb_intp count, cphvb_instruction inst_list[])
{
    if(count <= 0)
        return CPHVB_SUCCESS;
    
//    cphvb_pprint_instr_list(inst_list, count, "GLOBAL");

    for(cphvb_intp i=0; i < count; ++i)
    {
        cphvb_instruction* inst = &inst_list[i];
        assert(inst->opcode >= 0);
        switch(inst->opcode) 
        {
            case CPHVB_USERFUNC:
            {
                if (inst->userfunc->id == reduce_impl_id) 
                {
                    //TODO: the cphvb_reduce is hardcoded for now.
                    if(cphvb_reduce(inst->userfunc, NULL) != CPHVB_SUCCESS)
                        EXCEPT("[CLUSTER-VEM] The user-defined function cphvb_reduce failed.");
                }
                else
                {
                    fallback_exec(inst);
                }
                break;
            }
            case CPHVB_DISCARD:
            {
                cphvb_array *g_ary = inst->operand[0];
                if(g_ary->base == NULL)
                {
                    cphvb_array *l_ary = array_get_existing_local(g_ary);
                    if(l_ary != NULL)
                    {
                        batch_schedule(CPHVB_DISCARD, l_ary);
                    }
                }   
                dispatch_slave_known_remove(g_ary);
                break;
            }
            case CPHVB_FREE:
            {
                cphvb_array *g_ary = cphvb_base_array(inst->operand[0]);
                cphvb_array *l_ary = array_get_existing_local(g_ary);
                cphvb_data_free(g_ary);
                if(l_ary != NULL)
                    batch_schedule(CPHVB_FREE, l_ary);
                break;
            }
            case CPHVB_SYNC:
            {
                cphvb_array *base = cphvb_base_array(inst->operand[0]);
                comm_slaves2master(base);
                break;
            }
            case CPHVB_NONE:
            {
                break;
            }
            default:
            {
                execute_regular(inst);
            }
        }
    }
    
    //Lets flush all scheduled tasks
    batch_flush();
    //And remove all tmp data structures
    tmp_clear();

    return CPHVB_SUCCESS;
}



cphvb_error cphvb_reduce( cphvb_userfunc *arg, void* ve_arg)
{
    cphvb_reduce_type *a = (cphvb_reduce_type *) arg;   // Grab function arguments
    cphvb_opcode opcode = a->opcode;                    // Opcode
    cphvb_index axis    = a->axis;                      // The axis to reduce 

    return ufunc_reduce(opcode, axis, a->operand, reduce_impl_id);
}
