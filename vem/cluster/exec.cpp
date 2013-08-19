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
#include <bh.h>

#include "bh_vem_cluster.h"
#include "mapping.h"
#include "array.h"
#include "pgrid.h"
#include "dispatch.h"
#include "comm.h"
#include "except.h"
#include "ufunc_reduce.h"
#include "batch.h"
#include "tmp.h"
#include "timing.h"


//Function pointers to the Node VEM.
static bh_init vem_init;
static bh_shutdown vem_shutdown;
static bh_reg_func vem_reg_func;
//Public function pointer to the Node VEM
bh_execute exec_vem_execute;

//The VE components
static bh_component **my_components;

//Our self
static bh_component *myself;

//Number of user-defined functions registered.
static bh_intp userfunc_count = 0;
//User-defined function IDs.
static bh_userfunc_impl reduce_impl = NULL;
static bh_intp reduce_impl_id = 0;
static bh_userfunc_impl random_impl = NULL;
static bh_intp random_impl_id = 0;

//Number of instruction fallbacks
static bh_intp fallback_count = 0;


/* Initialize the VEM
 *
 * @return Error codes (BH_SUCCESS)
 */
bh_error exec_init(const char *component_name)
{
    bh_intp children_count;
    bh_error err;
    myself = bh_component_setup(component_name);
    if(myself == NULL)
        return BH_ERROR;

    err = bh_component_children(myself, &children_count, &my_components);
    if (children_count != 1)
    {
		std::cerr << "Unexpected number of child nodes for VEM, must be 1" << std::endl;
		return BH_ERROR;
    }

    if (err != BH_SUCCESS)
	    return err;

    vem_init = my_components[0]->init;
    exec_vem_execute = my_components[0]->execute;
    vem_shutdown = my_components[0]->shutdown;
    vem_reg_func = my_components[0]->reg_func;

    //Let us initiate the Node VEM.
    if((err = vem_init(my_components[0])) != 0)
        return err;

    return BH_SUCCESS;
}


/* Shutdown the VEM, which include a instruction flush
 *
 * @return Error codes (BH_SUCCESS)
 */
bh_error exec_shutdown(void)
{
    bh_error err;
    if((err = vem_shutdown()) != BH_SUCCESS)
        return err;
    bh_component_free(my_components[0]);//Only got one child.
    vem_init     = NULL;
    exec_vem_execute  = NULL;
    vem_shutdown = NULL;
    vem_reg_func = NULL;
    bh_component_free_ptr(my_components);
    my_components = NULL;

    //Finalize the process grid
    pgrid_finalize();

    //Finalize the process grid
    dispatch_finalize();

    if(fallback_count > 0)
        fprintf(stderr, "[CLUSTER-VEM] Warning - fallen back to "
        "sequential executing %ld times.\n", fallback_count);

    return BH_SUCCESS;
}


/* Register a new user-defined function.
 *
 * @lib Name of the shared library e.g. libmyfunc.so
 *      When NULL the default library is used.
 * @fun Name of the function e.g. myfunc
 * @id Identifier for the new function. The bridge should set the
 *     initial value to Zero. (in/out-put)
 * @return Error codes (BH_SUCCESS)
 */
bh_error exec_reg_func(char *fun, bh_intp *id)
{
    bh_error e;

    if(*id == 0)//Only if parent didn't set the ID.
    {
        *id = ++userfunc_count;
        assert(pgrid_myrank == 0);
    }

    if((e = vem_reg_func(fun, id)) != BH_SUCCESS)
    {
        *id = 0;
        return e;
    }

    //NB: For now all user-defined functions are hardcoded
    if(strcmp("bh_reduce", fun) == 0)
    {
        if(reduce_impl == NULL)
        {
            reduce_impl_id = *id;
            return BH_SUCCESS;
        }
    }
    else if(strcmp("bh_random", fun) == 0)
    {
        if(random_impl == NULL)
        {
            random_impl_id = *id;
            return BH_SUCCESS;
        }
    }

    return BH_SUCCESS;
}


/* Execute to instruction locally at the master-process
 *
 * @instruction The instructionto execute
 */
static void fallback_exec(bh_instruction *inst)
{
    int nop = bh_operands_in_instruction(inst);
    std::set<bh_base*> arys2discard;

    batch_flush();

    ++fallback_count;

    //Gather all data at the master-process
    bh_view *oprands = bh_inst_operands(inst);
    for(bh_intp o=0; o < nop; ++o)
    {
        bh_view *op = &oprands[o];
        if(bh_is_constant(op))
            continue;

        bh_base *base = bh_base_array(op);
        comm_slaves2master(base);
    }

    //Do global instruction
    if(pgrid_myrank == 0)
    {
        batch_schedule_inst(*inst);
    }
    batch_flush();

    //Scatter all data back to all processes
    for(bh_intp o=0; o < nop; ++o)
    {
        bh_view *op = &oprands[o];
        if(bh_is_constant(op))
            continue;
        bh_base *base = bh_base_array(op);

        //We have to make sure that the master-process has allocated memory
        //because the slaves cannot determine it.
        if(pgrid_myrank == 0)
            bh_data_malloc(base);

        comm_master2slaves(base);

        //All local arrays should be discarded
        arys2discard.insert(base);
    }
    //Free and discard all local base arrays
    for(std::set<bh_base*>::iterator it=arys2discard.begin();
        it != arys2discard.end(); ++it)
    {
        batch_schedule_inst(BH_FREE, *it);
        batch_schedule_inst(BH_DISCARD, *it);
    }
}


/* Execute a regular computation instruction
 *
 * @instruction The regular computation instruction
 */
static void execute_regular(bh_instruction *inst)
{
    std::vector<ary_chunk> chunks;
    int nop = bh_operands_in_instruction(inst);
    bh_view *operands = bh_inst_operands(inst);

    mapping_chunks(nop, operands, chunks);
    assert(chunks.size() > 0);

    //Handle one chunk at a time.
    for(std::vector<ary_chunk>::size_type c=0; c < chunks.size();c += nop)
    {
        assert(bh_nelements(chunks[0].ary.ndim, chunks[0].ary.shape) > 0);

        //The process where the output chunk is located will do the computation.
        int owner_rank = chunks[0+c].rank;

        //Create a local instruction based on the array-chunks
        bh_instruction local_inst = *inst;
        for(bh_intp k=0; k < nop; ++k)
        {
            if(!bh_is_constant(&inst->operand[k]))
            {
                ary_chunk *chunk = &chunks[k+c];
                local_inst.operand[k] = chunk->ary;
                comm_array_data(*chunk, owner_rank);
            }
        }

        //Check if we should do the computation
        if(pgrid_myrank != owner_rank)
            continue;

        //Schedule task
        batch_schedule_inst(local_inst);

        //Free and discard all local chunk arrays
        for(bh_intp k=0; k < nop; ++k)
        {
            if(bh_is_constant(&inst->operand[k]))
                continue;

            ary_chunk *chunk = &chunks[k+c];
            if(chunk->temporary)
            {
                batch_schedule_inst(BH_FREE, chunk->ary.base);
                batch_schedule_inst(BH_DISCARD, chunk->ary.base);
            }
        }
    }
}



/* Execute a list of instructions where all operands are global arrays
 *
 * @instruction A list of instructions to execute
 * @return Error codes
 */
bh_error exec_execute(bh_intp count, bh_instruction inst_list[])
{
    if(count <= 0)
        return BH_SUCCESS;

//    bh_pprint_instr_list(inst_list, count, "GLOBAL");
    bh_uint64 stime = bh_timing();

    for(bh_intp i=0; i < count; ++i)
    {
        bh_instruction* inst = &inst_list[i];
        assert(inst->opcode >= 0);
        switch(inst->opcode)
        {
            case BH_USERFUNC:
            {
                if (inst->userfunc->id == random_impl_id)
                {
                    //TODO: the bh_random is hardcoded for now.
                    if(bh_random(inst->userfunc, NULL) != BH_SUCCESS)
                        EXCEPT("[CLUSTER-VEM] The user-defined function bh_random failed.");
                }
                else
                {
                    fallback_exec(inst);
                }
                break;
            }
            case BH_ADD_REDUCE:
                ufunc_reduce(inst, BH_ADD);
                break;
            case BH_MULTIPLY_REDUCE:
                ufunc_reduce(inst, BH_MULTIPLY);
                break;
            case BH_MINIMUM_REDUCE:
                ufunc_reduce(inst, BH_MINIMUM);
                break;
            case BH_MAXIMUM_REDUCE:
                ufunc_reduce(inst, BH_MAXIMUM);
                break;
            case BH_LOGICAL_AND_REDUCE:
                ufunc_reduce(inst, BH_LOGICAL_AND);
                break;
            case BH_BITWISE_AND_REDUCE:
                ufunc_reduce(inst, BH_BITWISE_AND);
                break;
            case BH_LOGICAL_OR_REDUCE:
                ufunc_reduce(inst, BH_LOGICAL_OR);
                break;
            case BH_BITWISE_OR_REDUCE:
                ufunc_reduce(inst, BH_BITWISE_OR);
                break;
            case BH_DISCARD:
            {
                bh_base *g_ary = bh_base_array(&inst->operand[0]);
                bh_base *l_ary = array_get_existing_local(g_ary);
                if(l_ary != NULL)
                {
                    batch_schedule_inst(BH_DISCARD, l_ary);
                }
                dispatch_slave_known_remove(g_ary);
                break;
            }
            case BH_FREE:
            {
                bh_base *g_ary = bh_base_array(&inst->operand[0]);
                bh_base *l_ary = array_get_existing_local(g_ary);
                bh_data_free(g_ary);
                if(l_ary != NULL)
                    batch_schedule_inst(BH_FREE, l_ary);
                break;
            }
            case BH_SYNC:
            {
                bh_base *base = bh_base_array(&inst->operand[0]);
                comm_slaves2master(base);
                break;
            }
            case BH_NONE:
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

    bh_timing_save(timing_exec_execute, stime, bh_timing());
    return BH_SUCCESS;
}


bh_error bh_random( bh_userfunc *arg, void* ve_arg)
{
    bh_view *op = &arg->operand[0];

    std::vector<ary_chunk> chunks;
    mapping_chunks(1, op, chunks);
    assert(chunks.size() > 0);

    //Handle one chunk at a time.
    for(std::vector<ary_chunk>::size_type c=0; c < chunks.size(); ++c)
    {
        assert(bh_nelements(chunks[0].ary.ndim, chunks[0].ary.shape) > 0);
        //The process where the output chunk is located will do the computation.
        if(pgrid_myrank == chunks[c].rank)
        {
            bh_random_type *ufunc = (bh_random_type*)tmp_get_misc(sizeof(bh_random_type));
            ufunc->id          = random_impl_id;
            ufunc->nout        = 1;
            ufunc->nin         = 0;
            ufunc->struct_size = sizeof(bh_random_type);
            ufunc->operand[0]  = chunks[c].ary;
            batch_schedule_inst(BH_USERFUNC, NULL, (bh_userfunc*)(ufunc));
        }
    }
    return BH_SUCCESS;
}

