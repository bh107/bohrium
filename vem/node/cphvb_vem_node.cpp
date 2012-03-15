/*
 * Copyright 2011 Mads R. B. Kristensen <madsbk@gmail.com>
 *
 * This file is part of cphVB.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
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
static cphvb_com **coms;

//Our self
static cphvb_com *myself;

//Number of user-defined functions registered.
static cphvb_intp userfunc_count = 0;

#define PLAININST (1)
#define REDUCEINST (2)

ArrayManager* arrayManager;

/* Initialize the VEM
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error cphvb_vem_node_init(cphvb_com *self)
{
    cphvb_intp children_count;
    cphvb_error err;
    myself = self;

    cphvb_com_children(self, &children_count, &coms);
    ve_init = coms[0]->init;
    ve_execute = coms[0]->execute;
    ve_shutdown = coms[0]->shutdown;
    ve_reg_func = coms[0]->reg_func;

    //Let us initiate the simple VE and register what it supports.
    err = ve_init(coms[0]);
    if(err)
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
    cphvb_com_free(coms[0]);//Only got one child.
    free(coms);
    delete arrayManager;
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
 * @has_init_value Does the array have an initial value
 * @init_value The initial value
 * @new_array The handler for the newly created array
 * @return Error code (CPHVB_SUCCESS, CPHVB_OUT_OF_MEMORY)
 */
cphvb_error cphvb_vem_node_create_array(cphvb_array*   base,
                                        cphvb_type     type,
                                        cphvb_intp     ndim,
                                        cphvb_index    start,
                                        cphvb_index    shape[CPHVB_MAXDIM],
                                        cphvb_index    stride[CPHVB_MAXDIM],
                                        cphvb_intp     has_init_value,
                                        cphvb_constant init_value,
                                        cphvb_array**  new_array)
{

    try
    {
        *new_array = arrayManager->create(base, type, ndim, start, shape,
                                          stride, has_init_value, init_value);
    }
    catch(std::exception& e)
    {
        return CPHVB_OUT_OF_MEMORY;
    }
    #ifdef CPHVB_TRACE
        cphvb_com_trace_array(myself, *new_array);
    #endif

    return CPHVB_SUCCESS;
}

/* Registre a new user-defined function.
 *
 * @lib Name of the shared library e.g. libmyfunc.so
 *      When NULL the default library is used.
 * @fun Name of the function e.g. myfunc
 * @id Identifier for the new function. The bridge should set the
 *     initial value to Zero. (in/out-put)
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error cphvb_vem_node_reg_func(char *lib, char *fun, cphvb_intp *id)
{
    if(*id == 0)//Only if parent didn't set the ID.
        *id = ++userfunc_count;

    return ve_reg_func(lib, fun, id);
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
    cphvb_intp valid_instruction_count = count;
    for(i=0; i<count; ++i)
    {
        cphvb_instruction* inst = &inst_list[i];
        #ifdef CPHVB_TRACE
            cphvb_com_trace_inst(myself, inst);
        #endif
        switch(inst->opcode)
        {
        case CPHVB_DESTROY:
        {
            cphvb_array* base = cphvb_base_array(inst->operand[0]);
            if (inst->operand[0]->base != NULL)
            {   // It's a view and we can mark it for deletion
                arrayManager->erasePending(inst->operand[0]);
            }
            --base->ref_count; //decrease refcount
            if(base->ref_count <= 0)
            {
                // Mark the Base for deletion
                arrayManager->erasePending(base);
                if (base->owner != CPHVB_PARENT)
                {
                    //Tell the VE to discard the base array.
                    inst->operand[0] = base;
                    inst->opcode = CPHVB_DISCARD;
                }
                else
                {
                    inst->opcode = CPHVB_NONE;
                    --valid_instruction_count;
                }
            }
            else
            {   //Tell the VE to do nothing
                inst->opcode = CPHVB_NONE;
                --valid_instruction_count;
            }
            break;
        }
        case CPHVB_RELEASE:
        {
            cphvb_array* base = cphvb_base_array(inst->operand[0]);
            switch (base->owner)
            {
            case CPHVB_PARENT:
                //The owner is upstream so we do nothing
                inst->opcode = CPHVB_NONE;
                --valid_instruction_count;
                break;
            case CPHVB_SELF:
                //We own the data: Send discards down stream
                //and change owner to upstream
                inst->operand[0] = base;
                inst->opcode = CPHVB_DISCARD;
                arrayManager->changeOwnerPending(base,CPHVB_PARENT);
                break;
            default:
                //The owner is downstream so send the release down
                inst->operand[0] = base;
                arrayManager->changeOwnerPending(base,CPHVB_PARENT);
            }
            break;
        }
        case CPHVB_SYNC:
        {
            cphvb_array* base = cphvb_base_array(inst->operand[0]);
            switch (base->owner)
            {
            case CPHVB_PARENT:
            case CPHVB_SELF:
                //The owner is not down stream so we do nothing
                inst->opcode = CPHVB_NONE;
                --valid_instruction_count;
                break;
            default:
                //The owner is downstream so send the sync down
                //and take ownership
                inst->operand[0] = base;
                arrayManager->changeOwnerPending(base,CPHVB_SELF);
            }
            break;
        }
        case CPHVB_USERFUNC:
        {
            cphvb_userfunc *uf = inst->userfunc;
            //The children should own the output arrays.
            for(int i = 0; i < uf->nout; ++i)
            {
                cphvb_array* base = cphvb_base_array(uf->operand[i]);
                base->owner = CPHVB_CHILD;
            }
            //We should own the input arrays.
            for(int i = uf->nout; i < uf->nout + uf->nin; ++i)
            {
                cphvb_array* base = cphvb_base_array(uf->operand[i]);
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
            for (int i = 1; i < cphvb_operands(inst->opcode); ++i)
            {
                if(cphvb_base_array(inst->operand[i])->owner == CPHVB_PARENT)
                {
                    cphvb_base_array(inst->operand[i])->owner = CPHVB_SELF;
                }
            }
        }
        }
    }
    if (valid_instruction_count > 0)
    {
        cphvb_error e = ve_execute(count, inst_list);
        arrayManager->flush();
        return e;
    }
    else
    {
        // No valid instructions in batch
        return CPHVB_SUCCESS;
    }
}
