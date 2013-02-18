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

#include "bh_vem_node.h"

//Function pointers to the VE.
static bh_init ve_init;
static bh_execute ve_execute;
static bh_shutdown ve_shutdown;
static bh_reg_func ve_reg_func;

//The VE components
static bh_component **vem_node_components;

//Our self
static bh_component *vem_node_myself;

//Number of user-defined functions registered.
static bh_intp userfunc_count = 0;

//Allocated arrays
static std::set<bh_array*> allocated_arys;

//The timing ID for executions
static bh_intp exec_timing;


/* Initialize the VEM
 *
 * @return Error codes (BH_SUCCESS)
 */
bh_error bh_vem_node_init(bh_component *self)
{
    bh_intp children_count;
    bh_error err;
    vem_node_myself = self;

    err = bh_component_children(self, &children_count, &vem_node_components);
    if (children_count != 1) 
    {
		std::cerr << "Unexpected number of child nodes for VEM, must be 1" << std::endl;
		return BH_ERROR;
    }
    
    if (err != BH_SUCCESS)
	    return err;
    
    ve_init = vem_node_components[0]->init;
    ve_execute = vem_node_components[0]->execute;
    ve_shutdown = vem_node_components[0]->shutdown;
    ve_reg_func = vem_node_components[0]->reg_func;

    //Let us initiate the simple VE and register what it supports.
    if((err = ve_init(vem_node_components[0])) != 0)
        return err;

    exec_timing = bh_timing_new("node-execution");

    return BH_SUCCESS;
}


/* Shutdown the VEM, which include a instruction flush
 *
 * @return Error codes (BH_SUCCESS)
 */
bh_error bh_vem_node_shutdown(void)
{
    bh_error err;
    err = ve_shutdown();
    bh_component_free(vem_node_components[0]);//Only got one child.
    ve_init     = NULL;
    ve_execute  = NULL;
    ve_shutdown = NULL;
    ve_reg_func = NULL;
    bh_component_free_ptr(vem_node_components);
    vem_node_components = NULL;

    if(allocated_arys.size() > 0)
    {
        long s = (long) allocated_arys.size();
        if(s > 20)
            fprintf(stderr, "[NODE-VEM] Warning %ld arrays were not discarded "
                            "on exit (too many to show here).\n", s);
        else
        {
            fprintf(stderr, "[NODE-VEM] Warning %ld arrays were not discarded "
                            "on exit (only showing the array IDs because the "
                            "array list may be corrupted due to reuse of array structs):\n", s);
            for(std::set<bh_array*>::iterator it=allocated_arys.begin(); 
                it != allocated_arys.end(); ++it)
            {
                fprintf(stderr, "array id: %p\n", *it);
            }
        }
    }
    bh_timing_dump_all();
    return err;
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
bh_error bh_vem_node_reg_func(char *fun, bh_intp *id)
{
	bh_error e;
    
    if(*id == 0)//Only if parent didn't set the ID.
        *id = ++userfunc_count;

    if((e = ve_reg_func(fun, id)) != BH_SUCCESS)
    {
        *id = 0;
        return e;
    }
    return e;
}


/* Execute a list of instructions (blocking, for the time being).
 * It is required that the VEM supports all instructions in the list.
 *
 * @instruction A list of instructions to execute
 * @return Error codes (BH_SUCCESS)
 */
bh_error bh_vem_node_execute(bh_intp count,
                                   bh_instruction inst_list[])
{
    if (count <= 0)
        return BH_SUCCESS;

    bh_uint64 start = bh_timing();
    
    for(bh_intp i=0; i<count; ++i)
    {
        bh_instruction* inst = &inst_list[i];
        #ifdef BH_TRACE
            bh_component_trace_inst(vem_node_myself, inst);
        #endif
        int nop = bh_operands_in_instruction(inst);
        bh_array **operands = bh_inst_operands(inst);

        //Save all new arrays 
        for(bh_intp o=0; o<nop; ++o)
        {
            if(operands[o] != NULL)
                allocated_arys.insert(operands[o]);
        }

        //And remove discared arrays
        if(inst->opcode == BH_DISCARD)
        {
            bh_array *ary = operands[0];
            //Check that we are not discarding a base that still has views.
            if(ary->base == NULL)
            {
                for(std::set<bh_array*>::iterator it=allocated_arys.begin(); 
                    it != allocated_arys.end(); ++it)
                {
                    assert((*it)->base != ary);
                }
            }
            if(allocated_arys.erase(ary) != 1)
                fprintf(stderr, "[NODE-VEM] discarding unknown array\n");
        }               
    }

//    bh_pprint_instr_list(inst_list, count, "NODE");

    bh_error ret = ve_execute(count, inst_list);

    bh_timing_save(exec_timing, start, bh_timing());

    return ret;
}
