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

#ifndef __BH_COMPONENT_HPP
#define __BH_COMPONENT_HPP

#include <string>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include <bh_config_parser.hpp>
#include <bh_error.h>
#include <bh_ir.hpp>
#include <bh_opcode.h>

namespace bohrium {
namespace component {

/* A Component in Bohrium is a shared library that implement a specific functionality.
 * Requirements:
 *      - A component must be compiled into a shared library e.g. .so, .dylib, or .dll.
 *
 *	- A component has to implement the Implementation class; specifically,
 *	  the execute() method, which the Bridge calls when it wants to execute a BhIR,
 *	  and the extmethod() method, which the Bridge calls when it wants to register
 *	  a new extended method.
 *
 *      - A component has to implement two C compatible functions, create() and destroy(),
 *        which a parent component (or the bridge) can call to create or destroy an instance
 *        of the component. The Interface class makes it easy for a parent to retrieve and
 *        use these two functions.
 */

// Representation of a component implementation, which is a virtual class
// that all Bohrium components should implement
class Implementation {
  public:
    // The level in the runtime stack starting a zero, which is the bridge
    const unsigned int stack_level;
    // The configure file
    const ConfigParser config;
  public:
    Implementation(unsigned int stack_level) : stack_level(stack_level), config(stack_level) {};

    /* Execute a BhIR (graph of instructions)
     *
     * @bhir    The BhIR to execute
     * Throws exceptions on error
     */
    virtual void execute(bh_ir *bhir) = 0;

    /* Register a new extension method.
     *
     * @name   Name of the function e.g. matmul
     * @opcode Opcode for the new function.
     * Throws exceptions on error
     */
    virtual void extmethod(const std::string &name, bh_opcode opcode) = 0;
    virtual ~Implementation() {};
};

// Representation of a component interface, which consist of a create()
// and destroy() function.
class Interface {
  private:
    // Path to the shared library file e.g. .so, .dylib, or .dll
    std::string _lib_path;
    // Shared library handle
    void* _lib_handle;
    // Function pointer to the component's create function
    Implementation* (*_create)(unsigned int);
    // Function pointer to the component's destroy function
    void (*_destroy)(Implementation *component);
    Implementation *_implementation;
  public:
    // Constructor that takes the path to the shared library and
    // the stack level of the component
    Interface(const std::string &lib_path, unsigned int stack_level);
    ~Interface();
    // Get the component implementation
    Implementation* getImpl() { return _implementation; };
};

}} //namespace bohrium::component

#endif

