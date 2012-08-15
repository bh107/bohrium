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
#include <cphvb.h>

#include "cphvb_vem_node.h"
#include "ArrayManager.hpp"

//Function pointers to the VE.
static cphvb_init ve_init;
static cphvb_execute ve_execute;
static cphvb_shutdown ve_shutdown;
static cphvb_reg_func ve_reg_func;

//The VE components
static cphvb_component **vem_node_components;

//Our self
static cphvb_component *vem_node_myself;

//Number of user-defined functions registered.
static cphvb_intp vem_userfunc_count = 0;

ArrayManager* arrayManager;

/* Initialize the VEM
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error cphvb_vem_node_init(cphvb_component *self)
{
    cphvb_intp children_count;
    cphvb_error err;
    vem_node_myself = self;

    err = cphvb_component_children(self, &children_count, &vem_node_components);
    if (children_count != 1) 
    {
		std::cerr << "Unexpected number of child nodes for VEM, must be 1" << std::endl;
		return CPHVB_ERROR;
    }
    
    if (err != CPHVB_SUCCESS)
	    return err;
    
    ve_init = vem_node_components[0]->init;
    ve_execute = vem_node_components[0]->execute;
    ve_shutdown = vem_node_components[0]->shutdown;
    ve_reg_func = vem_node_components[0]->reg_func;

    //Let us initiate the simple VE and register what it supports.
    if((err = ve_init(vem_node_components[0])) != 0)
        return err;

    try
    {
        arrayManager = new ArrayManager();
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return CPHVB_ERROR;
    }

    return CPHVB_SUCCESS;
}


/* Shutdown the VEM, which include a instruction flush
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error cphvb_vem_node_shutdown(void)
{
    cphvb_error err;
    err = ve_shutdown();
    cphvb_component_free(vem_node_components[0]);//Only got one child.
    ve_init     = NULL;
    ve_execute  = NULL;
    ve_shutdown = NULL;
    ve_reg_func = NULL;
    cphvb_component_free_ptr(vem_node_components);
    vem_node_components = NULL;
    delete arrayManager;
    arrayManager = NULL;
    return err;
}


/* Create an array, which are handled by the VEM.
 *
 * @base Pointer to the base array. If NULL this is a base array
 * @type The type of data in the array
 * @ndim Number of dimensions
 * @start Index of the start element (always 0 for base-array)
 * @shape[CPHVB_MAXDIM] Number of elements in each dimention
 * @stride[CPHVB_MAXDIM] The stride for each dimention
 * @new_array The handler for the newly created array
 * @return Error code (CPHVB_SUCCESS, CPHVB_OUT_OF_MEMORY)
 */
cphvb_error cphvb_vem_node_create_array(cphvb_array*   base,
                                        cphvb_type     type,
                                        cphvb_intp     ndim,
                                        cphvb_index    start,
                                        cphvb_index    shape[CPHVB_MAXDIM],
                                        cphvb_index    stride[CPHVB_MAXDIM],
                                        cphvb_array**  new_array)
{
    try
    {
        *new_array = arrayManager->create(base, type, ndim, start, shape, stride);
    }
    catch(std::exception& e)
    {
        return CPHVB_OUT_OF_MEMORY;
    }
    #ifdef CPHVB_TRACE
        cphvb_component_trace_array(vem_node_myself, *new_array);
    #endif

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
cphvb_error cphvb_vem_node_reg_func(char *fun, cphvb_intp *id)
{
	cphvb_error e;
	cphvb_intp tmpid;
    
    if(*id == 0)//Only if parent didn't set the ID.
        tmpid = vem_userfunc_count + 1;

    e = ve_reg_func(fun, &tmpid);

    //If the call succeeded, register the id as taken and return it
    if (e == CPHVB_SUCCESS)
    {
	    if (tmpid > vem_userfunc_count)
	    	vem_userfunc_count = tmpid;
    	*id = tmpid;
    }
    
    return e;
}


/* Execute a list of instructions (blocking, for the time being).
 * It is required that the VEM supports all instructions in the list.
 *
 * @instruction A list of instructions to execute
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error cphvb_vem_node_execute(cphvb_intp count,
                                   cphvb_instruction inst_list[])
{
    cphvb_intp i;
    
	if (count <= 0)
		return CPHVB_SUCCESS;
    
    for(i=0; i<count; ++i)
    {
        cphvb_instruction* inst = &inst_list[i];
        #ifdef CPHVB_TRACE
            cphvb_component_trace_inst(vem_node_myself, inst);
        #endif
        switch(inst->opcode)
        {
        	//Special handling
			case CPHVB_FREE:
			case CPHVB_SYNC:
			case CPHVB_DISCARD:
			{
				switch(inst->opcode)
				{
					case CPHVB_FREE:
						assert(inst->operand[0]->base == NULL);
						arrayManager->freePending(inst);
						break;
					case CPHVB_DISCARD:
						arrayManager->erasePending(inst);
						break;
					case CPHVB_SYNC:
						arrayManager->changeOwnerPending(inst, cphvb_base_array(inst->operand[0]), CPHVB_SELF);
						break;
					default:
						assert(false);
				}
				break;
			}
			case CPHVB_USERFUNC:
			{
				cphvb_userfunc *uf = inst->userfunc;
				//The children should own the output arrays.
				for(int j = 0; j < uf->nout; ++j)
				{
					cphvb_array* base = cphvb_base_array(uf->operand[j]);
					base->owner = CPHVB_CHILD;
				}
				//We should own the input arrays.
				for(int j = uf->nout; j < uf->nout + uf->nin; ++j)
				{
					cphvb_array* base = cphvb_base_array(uf->operand[j]);
					if(base->owner == CPHVB_PARENT)
					{
						base->owner = CPHVB_SELF;
					}
				}
				break;
			}
			default:
			{
				cphvb_array* base = cphvb_base_array(inst->operand[0]);
				// "Regular" operation: set ownership and send down stream
				base->owner = CPHVB_CHILD;//The child owns the output ary.
				for (int j = 1; j < cphvb_operands(inst->opcode); ++j)
				{
					if(!cphvb_is_constant(inst->operand[j]) &&
					   cphvb_base_array(inst->operand[j])->owner == CPHVB_PARENT)
					{
						cphvb_base_array(inst->operand[j])->owner = CPHVB_SELF;
					}
				}
			}
        }
    }

	cphvb_error e1 = ve_execute(count, inst_list);
	cphvb_error e2 = arrayManager->flush();
	if (e1 != CPHVB_SUCCESS)
		return e1;
	else if (e2 != CPHVB_SUCCESS)
		return e2;
	else
		return CPHVB_SUCCESS;
}
