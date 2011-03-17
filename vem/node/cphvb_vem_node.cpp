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

#include <cphvb_vem_node.h>
#include <cphvb_vem.h>
#include <cphvb_ve.h>

#include "ArrayManager.hpp"

//Function pointers to the VE.
static cphvb_ve_init ve_init;
static cphvb_ve_execute ve_execute;
static cphvb_ve_shutdown ve_shutdown;

//For now, we determent which VE to use at compile time.
#ifdef CUDA
    #include <cphvb_ve_cuda.h>
#else
    #include <cphvb_ve_simple.h>
#endif

ArrayManager* arrayManager;

//The VE info.
cphvb_support ve_support;

/* Initialize the VEM
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error cphvb_vem_node_init(void)
{
    cphvb_intp opcode_count, type_count;
    cphvb_opcode opcode[CPHVB_MAX_NO_OPERANDS];
    cphvb_type type[CPHVB_NO_TYPES];
    cphvb_error err;

    //Find the VEM
    #ifdef CUDA
        ve_init = &cphvb_ve_cuda_init;
        ve_execute = &cphvb_ve_cuda_execute;
        ve_shutdown = &cphvb_ve_cuda_shutdown;
    #else
        ve_init = &cphvb_ve_simple_init;
        ve_execute = &cphvb_ve_simple_execute;
        ve_shutdown = &cphvb_ve_simple_shutdown;
    #endif

    //Let us initiate the simple VE and register what it supports.
    err = ve_init(&opcode_count, opcode, &type_count, type);
    if(err)
        return err;

    //Init to False.
    std::memset(ve_support.opcode, 0, CPHVB_NO_OPCODES*sizeof(cphvb_bool));
    std::memset(ve_support.type, 0, CPHVB_NO_TYPES*sizeof(cphvb_bool));

#ifdef DEBUG
    std::cout << "[VEM node] Supported opcodes:\n" << std::endl;
#endif
    while(--opcode_count >= 0)
    {
        ve_support.opcode[opcode[opcode_count]] = 1;//Set True
#ifdef DEBUG
        std::cout << "\t" << cphvb_opcode_text(opcode[opcode_count]) <<
            std::endl;
#endif
    }

#ifdef DEBUG
    std::cout << "[VEM node] Supported types:" << std::endl;
#endif
    while(--type_count >= 0)
    {
        ve_support.type[type[type_count]] = 1;//Set True
#ifdef DEBUG
        std::cout << "\t" << cphvb_type_text(type[type_count]) << std::endl;
#endif
    }

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
    return ve_shutdown();
}


/* Create an array, which are handled by the VEM.
 *
 * @base Pointer to the base array. If NULL this is a base array
 * @type The type of data in the array
 * @ndim Number of dimentions
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
    return CPHVB_SUCCESS;
}


/* Check whether the instruction is supported by the VEM or not
 *
 * @return non-zero when true and zero when false
 */
cphvb_intp cphvb_vem_node_instruction_check(cphvb_instruction *inst)
{
    switch(inst->opcode)
    {
    case CPHVB_DESTORY:
        return 1;
    case CPHVB_RELEASE:
        return 1;
    default:
        if(ve_support.opcode[inst->opcode])
        {
            cphvb_intp i;
            cphvb_intp nop = cphvb_operands(inst->opcode);
            for(i=0; i<nop; ++i)
            {
                cphvb_type t;
                if(inst->operand[i] == CPHVB_CONSTANT)
                    t = inst->const_type[i];
                else
                    t = inst->operand[i]->type;
                if(!ve_support.type[t])
                    return 0;
            }
            return 1;
        }
        else
            return 0;
    }
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
        cphvb_array* base = cphvb_base_array(inst->operand[0]);
        switch(inst->opcode)
        {
        case CPHVB_DESTORY:
            if (inst->operand[0]->base != NULL)
            {   // It's a view and we can mark it for deletion
                arrayManager->erasePending(inst->operand[0]);
            }
            if(--base->ref_count <= 0) //decrease refcount
            {
                // Mark the Base for deletion
                arrayManager->erasePending(base);
                //Tell the VE to discard the base array.
                inst->operand[0] = base;
                inst->opcode = CPHVB_DISCARD;
            }
            else
            {   //Tell the VE to do nothing
                inst->opcode = CPHVB_NONE;
                --valid_instruction_count;
            }
            break;
        case CPHVB_RELEASE:
            //Get the base
            if (base->owner == CPHVB_PARENT)
            {   //The owner is upstream so we do nothing
                inst->opcode = CPHVB_DISCARD;
            }
            else
            {
                //Tell the VE to release the array.
                inst->operand[0] = base;
                inst->opcode = CPHVB_RELEASE;
                arrayManager->changeOwnerPending(base,CPHVB_PARENT);
            }
            break;
        default:
            base->owner = CPHVB_CHILD;
        }
    }
    if (valid_instruction_count > 0)
    {
        return ve_execute(count, inst_list); 
    }
    else 
    {
#ifdef DEBUG
        std::cout << "[VEM node] No valid instructions in batch." << std::endl;
#endif
        return CPHVB_SUCCESS;
    }
}
