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

#ifndef __BH_INTERFACE_H
#define __BH_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <bh_type.h>
#include <bh_instruction.h>
#include <bh_opcode.h>
#include <bh_error.h>
#include <iniparser.h>
#include <bh_win.h>

//Maximum number of characters in the name of a component, a attribute or
//a function.
#define BH_COMPONENT_NAME_SIZE (1024)

//Maximum number of support childs for a component
#define BH_COMPONENT_MAX_CHILDS (10)

//Prototype of the bh_component datatype.
typedef struct bh_component_struct bh_component;

/* Initialize the component
 *
 * @return Error codes (BH_SUCCESS)
 */
typedef bh_error (*bh_init)(bh_component *self);


/* Shutdown the component, which include a instruction flush
 *
 * @return Error codes (BH_SUCCESS)
 */
typedef bh_error (*bh_shutdown)(void);


/* Execute a list of instructions (blocking, for the time being).
 * It is required that the component supports all instructions in the list.
 *
 * @instruction A list of instructions to execute
 * @return Error codes (BH_SUCCESS, BH_PARTIAL_SUCCESS)
 */
typedef bh_error (*bh_execute)(bh_intp count,
                                     bh_instruction inst_list[]);

/* Registre a new user-defined function.
 *
 * @lib Name of the shared library e.g. libmyfunc.so
 *      When NULL the default library is used.
 * @fun Name of the function e.g. myfunc
 * @id Identifier for the new function. The bridge should set the
 *     initial value to Zero. (in/out-put)
 * @return Error codes (BH_SUCCESS)
 */
typedef bh_error (*bh_reg_func)(char *fun,
                                      bh_intp *id);


/* User-defined function implementation.
 *
 * @arg Argument for the user-defined function implementation
 * @ve_arg Additional argument that can be added by the VE to accomidate 
 *         the specific implementation
 * @return Error codes (BH_SUCCESS)
 */
typedef bh_error (*bh_userfunc_impl)(bh_userfunc *arg, void* ve_arg);


/* Codes for known component types */
typedef enum
{
    BH_BRIDGE,
    BH_VEM,
    BH_VE,
    BH_COMPONENT_ERROR
}bh_component_type;


/* Data struct for the bh_component data type */
struct bh_component_struct
{
    char name[BH_COMPONENT_NAME_SIZE];
    dictionary *config;
    void *lib_handle;//Handle for the dynamic linked library.
    bh_component_type type;
    bh_init init;
    bh_shutdown shutdown;
    bh_execute execute;
    bh_reg_func reg_func;
};


/* Setup the root component, which normally is the bridge.
 *
 * @name The name of the root component. If NULL "bridge" 
         will be used.
 * @return The root component in the configuration.
 */
DLLEXPORT bh_component *bh_component_setup(const char* name);



/* Retrieves the children components of the parent.
 *
 * @parent The parent component (input).
 * @count Number of children components(output).
 * @children Array of children components (output).
 * NB: the array and all the children should be free'd by the caller.
 * @return Error code (BH_SUCCESS).
 */
DLLEXPORT bh_error bh_component_children(bh_component *parent, bh_intp *count,
                                               bh_component **children[]);


/* Frees the component.
 *
 * @return Error code (BH_SUCCESS).
 */
DLLEXPORT bh_error bh_component_free(bh_component *component);

/* Frees allocated data.
 *
 * @return Error code (BH_SUCCESS).
 */
DLLEXPORT bh_error bh_component_free_ptr(void* data);

/* Retrieves an user-defined function.
 *
 * @self The component.
 * @fun Name of the function e.g. myfunc
 * @ret_func Pointer to the function (output)
 *           Is NULL if the function doesn't exist
 * @return Error codes (BH_SUCCESS)
 */
DLLEXPORT bh_error bh_component_get_func(bh_component *self, char *func,
                                               bh_userfunc_impl *ret_func);

/* Trace an array creation.
 *
 * @self The component.
 * @ary  The array to trace.
 * @return Error code (BH_SUCCESS).
 */
DLLEXPORT bh_error bh_component_trace_array(bh_component *self, bh_array *ary);

/* Trace an instruction.
 *
 * @self The component.
 * @inst  The instruction to trace.
 * @return Error code (BH_SUCCESS).
 */
DLLEXPORT bh_error bh_component_trace_inst(bh_component *self, bh_instruction *inst);

/* Look up a key in the config file 
 *
 * @component The component.
 * @key       The key to lookup in the config file
 * @return    The value if found, otherwise NULL
 */
DLLEXPORT char* bh_component_config_lookup(bh_component *component, const char* key);

#ifdef __cplusplus
}
#endif

#endif
