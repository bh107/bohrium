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

#ifdef __cplusplus
extern "C" {
#endif

#include <cphvb_type.h>
#include <cphvb_instruction.h>
#include <cphvb_opcode.h>
#include <cphvb_error.h>
#include <iniparser.h>

//Maximum number of characters in the name of a component, a attribute or
//a function.
#define CPHVB_COM_NAME_SIZE (1024)

//Prototype of the cphvb_com datatype.
typedef struct cphvb_com_struct cphvb_com;

/* Initialize the component
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
typedef cphvb_error (*cphvb_init)(cphvb_intp *opcode_count,
                                  cphvb_opcode opcode_list[CPHVB_MAX_NO_OPERANDS],
                                  cphvb_intp *datatype_count,
                                  cphvb_type datatype_list[CPHVB_NO_TYPES],
                                  cphvb_com *self);


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

/* Codes for known component types */
typedef enum
{
    CPHVB_BRIDGE,
    CPHVB_VEM,
    CPHVB_VE,
    CPHVB_COMPONENT_ERROR
}cphvb_com_type;

struct cphvb_com_struct
{
    char name[CPHVB_COM_NAME_SIZE];
    dictionary *config;
    cphvb_com_type type;
    cphvb_init init;
    cphvb_shutdown shutdown;
    cphvb_execute execute;
    cphvb_create_array create_array; //Only for VEMs
};


/* Setup the root component, which normally is the bridge.
 *
 * @return A new component object.
 */
cphvb_com *cphvb_com_setup(void);


/* Retrieves the children components of the parent.
 *
 * @parent The parent component (input).
 * @count Number of children components(output).
 * @children Array of children components (output).
 * NB: the array and all the children should be free'd by the caller.
 * @return Error code (CPHVB_SUCCESS).
 */
cphvb_error cphvb_com_children(cphvb_com *parent, cphvb_intp *count,
                               cphvb_com **children[]);


/* Frees the component.
 *
 * @return Error code (CPHVB_SUCCESS).
 */
cphvb_error cphvb_com_free(cphvb_com *component);


#ifdef __cplusplus
}
#endif

#endif
