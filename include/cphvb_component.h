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
#define CPHVB_COMPONENT_NAME_SIZE (1024)

//Maximum number of support childs for a component
#define CPHVB_COMPONENT_MAX_CHILDS (10)

//Prototype of the cphvb_component datatype.
typedef struct cphvb_component_struct cphvb_component;

/* Initialize the component
 *
 * @return Error codes (CPHVB_SUCCESS)
 */
typedef cphvb_error (*cphvb_init)(cphvb_component *self);


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

/* Registre a new user-defined function.
 *
 * @lib Name of the shared library e.g. libmyfunc.so
 *      When NULL the default library is used.
 * @fun Name of the function e.g. myfunc
 * @id Identifier for the new function. The bridge should set the
 *     initial value to Zero. (in/out-put)
 * @return Error codes (CPHVB_SUCCESS)
 */
typedef cphvb_error (*cphvb_reg_func)(char *fun,
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
}cphvb_component_type;


/* Data struct for the cphvb_component data type */
struct cphvb_component_struct
{
    char name[CPHVB_COMPONENT_NAME_SIZE];
    dictionary *config;
    void *lib_handle;//Handle for the dynamic linked library.
    cphvb_component_type type;
    cphvb_init init;
    cphvb_shutdown shutdown;
    cphvb_execute execute;
    cphvb_reg_func reg_func;
};


/* Setup the root component, which normally is the bridge.
 *
 * @return A new component object.
 */
DLLEXPORT cphvb_component *cphvb_component_setup(void);


/* Retrieves the children components of the parent.
 *
 * @parent The parent component (input).
 * @count Number of children components(output).
 * @children Array of children components (output).
 * NB: the array and all the children should be free'd by the caller.
 * @return Error code (CPHVB_SUCCESS).
 */
DLLEXPORT cphvb_error cphvb_component_children(cphvb_component *parent, cphvb_intp *count,
                                               cphvb_component **children[]);


/* Frees the component.
 *
 * @return Error code (CPHVB_SUCCESS).
 */
DLLEXPORT cphvb_error cphvb_component_free(cphvb_component *component);

/* Frees allocated data.
 *
 * @return Error code (CPHVB_SUCCESS).
 */
DLLEXPORT cphvb_error cphvb_component_free_ptr(void* data);

/* Retrieves an user-defined function.
 *
 * @self The component.
 * @fun Name of the function e.g. myfunc
 * @ret_func Pointer to the function (output)
 *           Is NULL if the function doesn't exist
 * @return Error codes (CPHVB_SUCCESS)
 */
DLLEXPORT cphvb_error cphvb_component_get_func(cphvb_component *self, char *func,
                                               cphvb_userfunc_impl *ret_func);

/* Trace an array creation.
 *
 * @self The component.
 * @ary  The array to trace.
 * @return Error code (CPHVB_SUCCESS).
 */
DLLEXPORT cphvb_error cphvb_component_trace_array(cphvb_component *self, cphvb_array *ary);

/* Trace an instruction.
 *
 * @self The component.
 * @inst  The instruction to trace.
 * @return Error code (CPHVB_SUCCESS).
 */
DLLEXPORT cphvb_error cphvb_component_trace_inst(cphvb_component *self, cphvb_instruction *inst);

/* Look up a key in the config file 
 *
 * @component The component.
 * @key       The key to lookup in the config file
 * @return    The value if found, otherwise NULL
 */
DLLEXPORT char* cphvb_component_config_lookup(cphvb_component *component, const char* key);

#ifdef __cplusplus
}
#endif

#endif
