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
#include <bh_ir.hpp>
#include <bh_opcode.h>

namespace bohrium {
namespace component {

/* A Component in Bohrium is a shared library that implement a specific functionality.
 * Requirements:
 *  - A component must be compiled into a shared library e.g. .so, .dylib, or .dll.
 *
 *	- A component has to implement the ComponentImpl class; specifically,
 *	  the execute() method, which the Bridge calls when it wants to execute a BhIR,
 *	  and the extmethod() method, which the Bridge calls when it wants to register
 *	  a new extended method.
 *
 *  - A component has to implement two C compatible functions, create() and destroy(),
 *    which a parent component (or the bridge) can call to create or destroy an instance
 *    of the component. The ComponentFace class makes it easy for a parent to retrieve and
 *    use these two functions.
 */

// Representation of a component implementation, which is a virtual class
// that all Bohrium components should implement
class ComponentImpl {
  public:
    // The level in the runtime stack starting a -1, which is the bridge,
    // 0 is the first component in the stack list, 1 is the second component etc.
    const int stack_level;
    // The configure file
    const ConfigParser config;
    // Constructor
    ComponentImpl(int stack_level) : stack_level(stack_level), config(stack_level) {};
    virtual ~ComponentImpl() {}; // NB: a destructor implementation must exist

    /* Execute a BhIR (graph of instructions)
     *
     * @bhir  The BhIR to execute
     * Throws exceptions on error
     */
    virtual void execute(bh_ir *bhir) = 0;

    /* Register a new extension method.
     *
     * @name   Name of the function
     * @opcode Opcode for the new function.
     * Throws exceptions on error
     */
    virtual void extmethod(const std::string &name, bh_opcode opcode) = 0;

};

// Representation of a component interface, which consist of a create()
// and destroy() function.
class ComponentFace {
  private:
    // Shared library handle
    void* _lib_handle;
    // Function pointer to the component's create function
    ComponentImpl* (*_create)(unsigned int);
    // Function pointer to the component's destroy function
    void (*_destroy)(ComponentImpl *component);
    // Pointer to the implementation of the component
    ComponentImpl *_implementation;
  public:
    // Constructor that takes the path to the shared library and
    // the stack level of the component
    ComponentFace(const std::string &lib_path, int stack_level);
    ~ComponentFace();

    // No default, copy, or move constructor!
    ComponentFace() = delete;
    ComponentFace(const ComponentFace &other) = delete;
    ComponentFace(ComponentFace &&other) = delete;

    /* Execute a BhIR (graph of instructions)
     *
     * @bhir  The BhIR to execute
     * Throws exceptions on error
     */
    void execute(bh_ir *bhir) {
        assert(_implementation != NULL);
        _implementation->execute(bhir);
    };

    /* Register a new extension method.
     *
     * @name   Name of the function
     * @opcode Opcode for the new function.
     * Throws exceptions on error
     */
    void extmethod(const std::string &name, bh_opcode opcode){
        assert(_implementation != NULL);
        _implementation->extmethod(name, opcode);
    };
};

// Representation of a component implementation that has a child.
// This is purely for convenience, it adds a child interface and implement
// pass-through implementations of the required component methods.
class ComponentImplWithChild : public ComponentImpl {
protected:
    // The interface of the child
    ComponentFace child;
public:
    ComponentImplWithChild(int stack_level)
            : ComponentImpl(stack_level),
              child(ComponentImpl::config.getChildLibraryPath(), stack_level+1) {}
    virtual ~ComponentImplWithChild() {};
    virtual void execute(bh_ir *bhir) {
        child.execute(bhir);
    }
    virtual void extmethod(const std::string &name, bh_opcode opcode) {
        child.extmethod(name, opcode);
    };
};

}} //namespace bohrium::component

#endif
