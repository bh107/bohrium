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
#include <cphvb_win.h>

//Maximum number of characters in the name of a component, a attribute or
//a function.
#define CPHVB_COM_NAME_SIZE (1024)

//Maximum number of support childs for a component
#define CPHVB_COM_MAX_CHILDS (10)

//Prototype of the cphvb_com datatype.
typedef struct cphvb_com_struct cphvb_com;

/* Initialize the component
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
typedef cphvb_error (*cphvb_init)(cphvb_com *self);


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
 * @new_array The handler for the newly created array (output)
 * @return Error code (CPHVB_SUCCESS, CPHVB_OUT_OF_MEMORY)
 */
typedef cphvb_error (*cphvb_create_array)(cphvb_array*   base,
                                          cphvb_type     type,
                                          cphvb_intp     ndim,
                                          cphvb_index    start,
                                          cphvb_index    shape[CPHVB_MAXDIM],
                                          cphvb_index    stride[CPHVB_MAXDIM],
                                          cphvb_array**  new_array);

/* Registre a new user-defined function.
 *
 * @lib Name of the shared library e.g. libmyfunc.so
 *      When NULL the default library is used.
 * @fun Name of the function e.g. myfunc
 * @id Identifier for the new function. The bridge should set the
 *     initial value to Zero. (in/out-put)
 * @return Error codes (CPHVB_SUCCESS)
 */
typedef cphvb_error (*cphvb_reg_func)(char *lib, char *fun,
                                      cphvb_intp *id);


/* User-defined function implementation.
 *
 * @arg Argument for the user-defined function implementation
 * @ve_arg Additional argument that can be added by the VE to accomidate 
 *         the specific implementation
 * @return Error codes (CPHVB_SUCCESS)
 */
typedef cphvb_error (*cphvb_userfunc_impl)(cphvb_userfunc *arg, void* ve_arg);


/* Codes for known component types */
typedef enum
{
    CPHVB_BRIDGE,
    CPHVB_VEM,
    CPHVB_VE,
    CPHVB_COMPONENT_ERROR
}cphvb_com_type;


/* Data struct for the cphvb_com data type */
struct cphvb_com_struct
{
    char name[CPHVB_COM_NAME_SIZE];
    dictionary *config;
    void *lib_handle;//Handle for the dynamic linked library.
    cphvb_com_type type;
    cphvb_init init;
    cphvb_shutdown shutdown;
    cphvb_execute execute;
    cphvb_reg_func reg_func;
    cphvb_create_array create_array; //Only for VEMs
};


/* Setup the root component, which normally is the bridge.
 *
 * @return A new component object.
 */
DLLEXPORT cphvb_com *cphvb_com_setup(void);


/* Retrieves the children components of the parent.
 *
 * @parent The parent component (input).
 * @count Number of children components(output).
 * @children Array of children components (output).
 * NB: the array and all the children should be free'd by the caller.
 * @return Error code (CPHVB_SUCCESS).
 */
DLLEXPORT cphvb_error cphvb_com_children(cphvb_com *parent, cphvb_intp *count,
                               cphvb_com **children[]);


/* Frees the component.
 *
 * @return Error code (CPHVB_SUCCESS).
 */
DLLEXPORT cphvb_error cphvb_com_free(cphvb_com *component);

/* Frees allocated data.
 *
 * @return Error code (CPHVB_SUCCESS).
 */
DLLEXPORT cphvb_error cphvb_com_free_ptr(void* data);

/* Retrieves an user-defined function.
 *
 * @self The component.
 * @lib Name of the shared library e.g. libmyfunc.so
*       When NULL the default library is used.
 * @fun Name of the function e.g. myfunc
 * @ret_func Pointer to the function (output)
 *           Is NULL if the function doesn't exist
 * @return Error codes (CPHVB_SUCCESS)
 */
DLLEXPORT cphvb_error cphvb_com_get_func(cphvb_com *self, char *lib, char *func,
                               cphvb_userfunc_impl *ret_func);

/* Trace an array creation.
 *
 * @self The component.
 * @ary  The array to trace.
 * @return Error code (CPHVB_SUCCESS).
 */
DLLEXPORT cphvb_error cphvb_com_trace_array(cphvb_com *self, cphvb_array *ary);

/* Trace an instruction.
 *
 * @self The component.
 * @inst  The instruction to trace.
 * @return Error code (CPHVB_SUCCESS).
 */
DLLEXPORT cphvb_error cphvb_com_trace_inst(cphvb_com *self, cphvb_instruction *inst);

#ifdef __cplusplus
}
#endif

#endif
