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

#ifndef __CPHVB_VEM_NODE_H
#define __CPHVB_VEM_NODE_H

#include <cphvb.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the VEM
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
DLLEXPORT cphvb_error cphvb_vem_node_init(cphvb_component *self);


/* Shutdown the VEM, which include a instruction flush
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
DLLEXPORT cphvb_error cphvb_vem_node_shutdown(void);


/* Create an array, which are handled by the VEM.
 *
 * @base Pointer to the base array. If NULL this is a base array
 * @type The type of data in the array
 * @ndim Number of dimentions
 * @start Index of the start element (always 0 for base-array)
 * @shape[CPHVB_MAXDIM] Number of elements in each dimention
 * @stride[CPHVB_MAXDIM] The stride for each dimention
 * @new_array The handler for the newly created array
 * @return Error code (CPHVB_SUCCESS, CPHVB_OUT_OF_MEMORY)
 */
DLLEXPORT cphvb_error cphvb_vem_node_create_array(cphvb_array*   base,
                                        cphvb_type     type,
                                        cphvb_intp     ndim,
                                        cphvb_index    start,
                                        cphvb_index    shape[CPHVB_MAXDIM],
                                        cphvb_index    stride[CPHVB_MAXDIM],
                                        cphvb_array**  new_array);


/* Execute a list of instructions (blocking, for the time being).
 * It is required that the VEM supports all instructions in the list.
 *
 * @instruction A list of instructions to execute
 * @return Error codes (CPHVB_SUCCESS)
 */
DLLEXPORT cphvb_error cphvb_vem_node_execute(cphvb_intp count,
                                   cphvb_instruction inst_list[]);

/* Registre a new user-defined function.
 *
 * @lib Name of the shared library e.g. libmyfunc.so
 *      When NULL the default library is used.
 * @fun Name of the function e.g. myfunc
 * @id Identifier for the new function. The bridge should set the
 *     initial value to Zero. (in/out-put)
 * @return Error codes (CPHVB_SUCCESS)
 */
DLLEXPORT cphvb_error cphvb_vem_node_reg_func(char *fun, cphvb_intp *id);

#ifdef __cplusplus
}
#endif

#endif
