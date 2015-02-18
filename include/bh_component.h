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


#include <bh_type.h>
#include <bh_instruction.h>
#include <bh_opcode.h>
#include <bh_error.h>
#include <bh_iniparser.h>
#include <bh_win.h>
#include <bh_ir.h>


#ifdef __cplusplus
extern "C" {
#endif

//Maximum number of characters in the name of a component, a attribute or
//a function.
#define BH_COMPONENT_NAME_SIZE (1024)

//Maximum number of support childs for a component
#define BH_COMPONENT_MAX_CHILDS (6)

/* Initialize the component
 *
 * @name    The name of the component e.g. node
 * @return  Error codes (BH_SUCCESS, BH_ERROR, BH_OUT_OF_MEMORY)
 */
typedef bh_error (*bh_init)(const char *name);

/* Shutdown the component, which include a instruction flush
 *
 * @return Error codes (BH_SUCCESS, BH_ERROR)
 */
typedef bh_error (*bh_shutdown)(void);

/* Execute a BhIR (graph of instructions)
 *
 * @bhir    The BhIR to execute
 * @return  Error codes (BH_SUCCESS, BH_ERROR, BH_OUT_OF_MEMORY)
 */
typedef bh_error (*bh_execute)(bh_ir* bhir);

/* Register a new extension method.
 *
 * @name   Name of the function e.g. matmul
 * @opcode Opcode for the new function.
 * @return Error codes (BH_SUCCESS, BH_ERROR, BH_OUT_OF_MEMORY,
 *                      BH_EXTMETHOD_NOT_SUPPORTED)
 */
typedef bh_error (*bh_extmethod)(const char *name, bh_opcode opcode);

/* Extension method prototype implementation.
 *
 * @instr  The extension method instruction to handle
 * @arg    Additional component specific argument
 * @return Error codes (BH_SUCCESS, BH_ERROR, BH_OUT_OF_MEMORY,
 *                      BH_TYPE_NOT_SUPPORTED)
 */
typedef bh_error (*bh_extmethod_impl)(bh_instruction *instr, void* arg);


/* The interface functions of a component */
typedef struct
{
    //Name of the component
    char name[BH_COMPONENT_NAME_SIZE];
    //Handle for the dynamic linked library.
    void *lib_handle;
    //The interface function pointers
    bh_init       init;
    bh_shutdown   shutdown;
    bh_execute    execute;
    bh_extmethod  extmethod;
}bh_component_iface;


/* Codes for known component types */
typedef enum
{
    BH_BRIDGE,
    BH_VEM,
    BH_VE,
    BH_FILTER,
    BH_FUSER,
    BH_COMPONENT_ERROR
}bh_component_type;


/* The component object */
typedef struct
{
    //Name of the component
    char name[BH_COMPONENT_NAME_SIZE];
    //The ini-config dictionary
    dictionary *config;
    //The component type
    bh_component_type type;
    //Number of children
    bh_intp nchildren;
    //The interface of the children of this component
    bh_component_iface children[BH_COMPONENT_MAX_CHILDS];
}bh_component;

/* Initilize the component object
 *
 * @self   The component object to initilize
 * @name   The name of the component. If NULL "bridge" will be used.
 * @return Error codes (BH_SUCCESS, BH_ERROR)
 */
DLLEXPORT bh_error bh_component_init(bh_component *self, const char* name);

/* Destroyes the component object.
 *
 * @self   The component object to destroy
 */
DLLEXPORT void bh_component_destroy(bh_component *self);

/* Retrieves an extension method implementation.
 *
 * @self      The component object.
 * @name      Name of the extension method e.g. matmul
 * @extmethod Pointer to the method (output)
 * @return    Error codes (BH_SUCCESS, BH_ERROR, BH_OUT_OF_MEMORY,
 *                         BH_EXTMETHOD_NOT_SUPPORTED)
 */
DLLEXPORT bh_error bh_component_extmethod(const bh_component *self,
                                          const char *name,
                                          bh_extmethod_impl *extmethod);

/* Look up a key in the config file
 *
 * @component The component.
 * @key       The key to lookup in the config file
 * @return    The value if found, otherwise NULL
 */
DLLEXPORT char* bh_component_config_lookup(const bh_component *component,
                                           const char* key);
DLLEXPORT int bh_component_config_lookup_bool(const bh_component *component,
                                              const char* key, int notfound);
DLLEXPORT int bh_component_config_lookup_int(const bh_component *component,
                                             const char* key, int notfound);
DLLEXPORT double bh_component_config_lookup_double(const bh_component *component,
                                                   const char* key, double notfound);

#ifdef __cplusplus
}
#endif

#endif

