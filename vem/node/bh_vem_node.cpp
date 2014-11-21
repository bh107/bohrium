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
#include <bh.h>
#include <set>
#define BH_TIMING_SUM
#include <bh_timing.hpp>

#include "bh_vem_node.h"


//Function pointers to our child.
static bh_component_iface *child;

//Our self
static bh_component vem_node_myself;

//Allocated base arrays
static std::set<bh_base*> allocated_bases;

//The timing ID for executions
static bh_intp exec_timing;

static int timing;

/* Component interface: init (see bh_component.h) */
bh_error bh_vem_node_init(const char* name)
{
    bh_error err;

    if((err = bh_component_init(&vem_node_myself, name)) != BH_SUCCESS)
        return err;

    timing = bh_component_config_lookup_bool(&vem_node_myself, "timimg", 0);
    
    //For now, we have one child exactly
    if(vem_node_myself.nchildren != 1)
    {
        std::cerr << "[NODE-VEM] Unexpected number of children, must be 1" << std::endl;
        return BH_ERROR;
    }

    //Let us initiate the child.
    child = &vem_node_myself.children[0];
    if((err = child->init(child->name)) != 0)
        return err;

    if (timing)
        exec_timing = bh_timer_new("[Node] Execution");

    return BH_SUCCESS;
}

/* Component interface: shutdown (see bh_component.h) */
bh_error bh_vem_node_shutdown(void)
{
    bh_error err = child->shutdown();
    bh_component_destroy(&vem_node_myself);

    if(allocated_bases.size() > 0)
    {
        long s = (long) allocated_bases.size();
        if(s > 20)
            fprintf(stderr, "[NODE-VEM] Warning %ld base arrays were not discarded "
                            "on exit (too many to show here).\n", s);
        else
        {
            fprintf(stderr, "[NODE-VEM] Warning %ld base arrays were not discarded "
                            "on exit (only showing the array IDs because the "
                            "view list may be corrupted due to reuse of base structs):\n", s);
            for(std::set<bh_base*>::iterator it=allocated_bases.begin();
                it != allocated_bases.end(); ++it)
            {
                fprintf(stderr, "base id: %p\n", *it);
            }
        }
    }
    if (timing)
        bh_timer_finalize(exec_timing);

    return err;
}

/* Component interface: extmethod (see bh_component.h) */
bh_error bh_vem_node_extmethod(const char *name, bh_opcode opcode)
{
    return child->extmethod(name, opcode);
}

//Inspect one instruction
static bh_error inspect(bh_instruction *instr)
{
    int nop = bh_operands_in_instruction(instr);
    bh_view *operands = bh_inst_operands(instr);

    //Save all new base arrays
    for(bh_intp o=0; o<nop; ++o)
    {
        if(!bh_is_constant(&operands[o]))
            allocated_bases.insert(operands[o].base);
    }

    //And remove discared arrays
    if(instr->opcode == BH_DISCARD)
    {
        bh_base *base = operands[0].base;
        if(allocated_bases.erase(base) != 1)
        {
            fprintf(stderr, "[NODE-VEM] discarding unknown base array (%p)\n", base);
            return BH_ERROR;
        }
    }
    return BH_SUCCESS;
}

/* Component interface: execute (see bh_component.h) */
bh_error bh_vem_node_execute(bh_ir* bhir)
{
    bh_uint64 start;
    if (timing)
        start = bh_timer_stamp();

    //Inspect the BhIR for new base arrays starting at the root DAG
    bh_ir_map_instr(bhir, &bhir->dag_list[0], &inspect);

    bh_error ret = child->execute(bhir);
    
    if (timing)
        bh_timer_add(exec_timing, start, bh_timer_stamp());

    return ret;
}
