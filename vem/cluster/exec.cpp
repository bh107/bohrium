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
#include <map>

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
#include "exec.h"

//Our self
static bh_component myself;

//Function pointers to our child.
bh_component_iface *mychild;

//Known extension methods
static std::map<bh_opcode, bh_extmethod_impl> extmethod_op2impl;

//Number of instruction fallbacks
static bh_intp fallback_count = 0;

/* Component interface: init (see bh_component.h) */
bh_error exec_init(const char *component_name)
{
    bh_error err;

    if((err = bh_component_init(&myself, component_name)) != BH_SUCCESS)
        return err;

    //For now, we have one child exactly
    if(myself.nchildren != 1)
    {
        std::cerr << "[CLUSTER-VEM] Unexpected number of children, must be 1" << std::endl;
        return BH_ERROR;
    }

    //Let us initiate the child.
    mychild = &myself.children[0];
    if((err = mychild->init(mychild->name)) != 0)
        return err;

    return BH_SUCCESS;
}

/* Component interface: shutdown (see bh_component.h) */
bh_error exec_shutdown(void)
{
    bh_error err;
    if((err = mychild->shutdown()) != BH_SUCCESS)
        return err;
    bh_component_destroy(&myself);

    //Finalize the process grid
    pgrid_finalize();

    //Finalize the process grid
    dispatch_finalize();

    if(fallback_count > 0)
        fprintf(stderr, "[CLUSTER-VEM] Warning - fallen back to "
        "sequential executing %ld times.\n", fallback_count);

    return BH_SUCCESS;
}

/* Component interface: extmethod (see bh_component.h) */
bh_error exec_extmethod(const char *name, bh_opcode opcode)
{
    bh_extmethod_impl extmethod;
    bh_error err = bh_component_extmethod(&myself, name, &extmethod);
    if(err != BH_SUCCESS)
        return err;

    if(extmethod_op2impl.find(opcode) != extmethod_op2impl.end())
    {
        printf("[CLUSTER-VEM] Warning, multiple registrations of the same"
               "extension method '%s' (opcode: %d)\n", name, (int)opcode);
    }
    extmethod_op2impl[opcode] = extmethod;
    return BH_SUCCESS;
}

/* Execute a regular computation instruction
 *
 * @instruction The regular computation instruction
 */
static void execute_regular(bh_instruction *inst)
{
    std::vector<ary_chunk> chunks;
    int nop = bh_noperands(inst->opcode);
    bh_view *operands = bh_inst_operands(inst);

    mapping_chunks(nop, operands, chunks);
    if(chunks.size() == 0)
        return;

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

        //BH_RANGE is a special case where we have to add the chunk offset to
        //the local arrays (excl. the rank 0 chunk)
        if(inst->opcode == BH_RANGE && pgrid_myrank > 0)
        {
            bh_instruction i;
            i.opcode = BH_ADD;
            i.operand[0] = chunks[c].ary;
            i.operand[1] = chunks[c].ary;
            i.operand[2] = chunks[c].ary;
            bh_flag_constant(&i.operand[2]);
            i.constant.value.uint64 = chunks[c].coord[0];
            i.constant.type = BH_UINT64;
            batch_schedule_inst(i);
        }

        //Free all local chunk arrays
        for(bh_intp k=0; k < nop; ++k)
        {
            if(bh_is_constant(&inst->operand[k]))
                continue;

            ary_chunk *chunk = &chunks[k+c];
            if(chunk->temporary)
            {
                batch_schedule_inst_on_base(BH_FREE, chunk->ary.base);
            }
        }
    }
}

/* Execute a single global instruction where all operands are global arrays
 *
 * @instr  The instruction in question that
 * @return Error codes
 */
static bh_error execute_instr(bh_instruction *inst)
{
    assert(inst->opcode >= 0);
    switch(inst->opcode)
    {
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
        case BH_FREE:
        {
            bh_base *g_ary = bh_base_array(&inst->operand[0]);
            bh_base *l_ary = array_get_existing_local(g_ary);
            bh_data_free(g_ary);
            if(l_ary != NULL) {
                batch_schedule_inst_on_base(BH_FREE, l_ary);
            }
            dispatch_slave_known_remove(g_ary);
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
    return BH_SUCCESS;
}


/* Execute a BhIR where all operands are global arrays
 *
 * @bhir   The BhIR in question
 * @return Error codes
 */
bh_error exec_execute(bh_ir *bhir)
{
//    bh_pprint_instr_list(inst_list, count, "GLOBAL");
    bh_uint64 stime = bh_timer_stamp();

    //Execute each instruction in the BhIR
    for(uint64_t i=0; i < bhir->instr_list.size(); ++i)
        execute_instr(&bhir->instr_list[i]);

    //Lets flush all scheduled tasks
    batch_flush();
    //And remove all tmp data structures
    tmp_clear();

    bh_timer_add(timing_exec_execute, stime, bh_timer_stamp());
    return BH_SUCCESS;
}
