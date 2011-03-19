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

#ifndef __CPHVB_VEM_H
#define __CPHVB_VEM_H

#include <cphvb_type.h>
#include <cphvb_instruction.h>
#include <cphvb_opcode.h>


typedef struct
{
    cphvb_bool opcode[CPHVB_NO_OPCODES];//list of opcode support
    cphvb_bool type[CPHVB_NO_OPCODES];  //list of type support
} cphvb_support;


/* Codes for known components */
enum /* cphvb_comp */
{
    CPHVB_PARENT,
    CPHVB_SELF,
    CPHVB_CHILD

};
typedef cphvb_intp cphvb_comp;


/* Initialize the VEM
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
typedef cphvb_error (*cphvb_vem_init)(void);


/* Shutdown the VEM, which include a instruction flush
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
typedef cphvb_error (*cphvb_vem_shutdown)(void);


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
typedef cphvb_error (*cphvb_vem_create_array)(
                                   cphvb_array*   base,
                                   cphvb_type     type,
                                   cphvb_intp     ndim,
                                   cphvb_index    start,
                                   cphvb_index    shape[CPHVB_MAXDIM],
                                   cphvb_index    stride[CPHVB_MAXDIM],
                                   cphvb_intp     has_init_value,
                                   cphvb_constant init_value,
                                   cphvb_array**  new_array);


/* Check whether the instruction is supported by the VEM or not
 *
 * @return non-zero when true and zero when false
 */
typedef cphvb_intp (*cphvb_vem_instruction_check)(cphvb_instruction *inst);

/* Execute a list of instructions (blocking, for the time being).
 * It is required that the VEM supports all instructions in the list.
 *
 * @instruction A list of instructions to execute
 * @return Error codes (CPHVB_SUCCESS)
 */
typedef cphvb_error (*cphvb_vem_execute)(cphvb_intp count,
                                         cphvb_instruction inst_list[]);


#endif
