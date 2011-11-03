/*
 * Copyright 2011 Mads R. B. Kristensen <madsbk@gmail.com>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
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

#ifndef __CPHVB_INTERFACE_H
#define __CPHVB_INTERFACE_H

#include <cphvb_type.h>
#include <cphvb_instruction.h>
#include <cphvb_opcode.h>


/* Initialize the component
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
typedef cphvb_error (*cphvb_init)(cphvb_intp *opcode_count,
                                  cphvb_opcode opcode_list[CPHVB_MAX_NO_OPERANDS],
                                  cphvb_intp *datatype_count,
                                  cphvb_type datatype_list[CPHVB_NO_TYPES]);


/* Shutdown the component, which include a instruction flush
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
typedef cphvb_error (*cphvb_shutdown)(void);


/* Execute a list of instructions (blocking, for the time being).
 * It is required that the component supports all instructions in the list.
 *
 * @instruction A list of instructions to execute
 * @return Error codes (CPHVB_SUCCESS, CPHVB_PARTIAL_SUCCESS)
 */
typedef cphvb_error (*cphvb_execute)(cphvb_intp count,
                                     cphvb_instruction inst_list[]);

/* Create an array, which are handled by the component.
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
typedef cphvb_error (*cphvb_create_array)(
                                   cphvb_array*   base,
                                   cphvb_type     type,
                                   cphvb_intp     ndim,
                                   cphvb_index    start,
                                   cphvb_index    shape[CPHVB_MAXDIM],
                                   cphvb_index    stride[CPHVB_MAXDIM],
                                   cphvb_intp     has_init_value,
                                   cphvb_constant init_value,
                                   cphvb_array**  new_array);


/* Check whether the instruction is supported by the component or not
 *
 * @return non-zero when true and zero when false
 */
typedef cphvb_intp (*cphvb_instruction_check)(cphvb_instruction *inst);


/* Codes for known components */
typedef enum
{
    CPHVB_BRIDGE,
    CPHVB_VEM,
    CPHVB_VE,
    CPHVB_COMPONENT_ERROR
}cphvb_component;

typedef struct
{
    cphvb_component type;
    cphvb_init init;
    cphvb_shutdown shutdown;
    cphvb_execute execute;
    cphvb_create_array create_array; //Only for VEMs
    cphvb_instruction_check instruction_check; //Only for VEMs
} cphvb_interface;

#endif
