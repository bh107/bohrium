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
    virtual void execute(BhIR *bhir) = 0;

    /* Register a new extension method.
     *
     * @name   Name of the function
     * @opcode Opcode for the new function.
     * Throws exceptions on error
     */
    virtual void extmethod(const std::string &name, bh_opcode opcode) = 0;

    /* Send and receive a message through the component stack
     *
     * @msg    The message to send
     * @return The received message
     * Throws exceptions on error
     */
    virtual std::string message(const std::string &msg) = 0;

    /* Get data pointer from the first VE in the runtime stack
     * NB: this doesn't include a flush
     *
     * @base         The base array that owns the data
     * @copy2host    Always copy the memory to main memory
     * @force_alloc  Force memory allocation
     * @nullify      Set the data pointer to NULL after returning
     * @return       The data pointer (NB: might point to device memory)
     * Throws exceptions on error
     */
    virtual void* get_mem_ptr(bh_base &base, bool copy2host, bool force_alloc, bool nullify) = 0;

    /* Set data pointer in the first VE in the runtime stack
     * NB: The component will deallocate the memory when encountering a BH_FREE.
     *     Also, this doesn't include a flush
     *
     * @base      The base array that will own the data
     * @host_ptr  The pointer points to the host memory (main memory) as opposed to device memory
     * @mem       The data pointer
     * Throws exceptions on error
     */
    virtual void set_mem_ptr(bh_base *base, bool host_ptr, void *mem) = 0;

    /* Get the device handle, such as OpenCL's cl_context, of the first VE in the runtime stack.
     * If the first VE isn't a device, NULL is returned.
     *
     * @return  The device handle
     * Throws exceptions on error
     */
    virtual void* get_device_context() = 0;

    /* Set the device context, such as CUDA's context, of the first VE in the runtime stack.
     * If the first VE isn't a device, nothing happens
     *
     * @device_context  The new device context
     * Throws exceptions on error
     */
    virtual void set_device_context(void* device_context) = 0;
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

    void execute(BhIR *bhir) {
        assert(_implementation != NULL);
        _implementation->execute(bhir);
    };
    void extmethod(const std::string &name, bh_opcode opcode) {
        assert(_implementation != NULL);
        _implementation->extmethod(name, opcode);
    };
    std::string message(const std::string &msg) {
        assert(_implementation != NULL);
        return _implementation->message(msg);
    }
    void* get_mem_ptr(bh_base &base, bool copy2host, bool force_alloc, bool nullify) {
        assert(_implementation != NULL);
        return _implementation->get_mem_ptr(base, copy2host, force_alloc, nullify);
    }
    virtual void set_mem_ptr(bh_base *base, bool host_ptr, void *mem) {
        assert(_implementation != NULL);
        return _implementation->set_mem_ptr(base, host_ptr, mem);
    }
    void* get_device_context() {
        assert(_implementation != NULL);
        return _implementation->get_device_context();
    };
    void set_device_context(void* device_context) {
        assert(_implementation != NULL);
        _implementation->set_device_context(device_context);
    };
};

// Representation of a component implementation that has a child.
// This is purely for convenience, it adds a child interface and implement
// pass-through implementations of the required component methods.
class ComponentImplWithChild : public ComponentImpl {
protected:
    // The interface of the child
    ComponentFace child;
    // Flag that indicate whether the component is enabled or disabled.
    // When disabled, the component should pass through instructions untouched to its child
    bool disabled;
public:
    ComponentImplWithChild(int stack_level)
            : ComponentImpl(stack_level),
              child(ComponentImpl::config.getChildLibraryPath(), stack_level+1), disabled(false) {}
    virtual ~ComponentImplWithChild() {};
    virtual void execute(BhIR *bhir) {
        child.execute(bhir);
    }
    virtual void extmethod(const std::string &name, bh_opcode opcode) {
        child.extmethod(name, opcode);
    };
    virtual std::string message(const std::string &msg) {
        return child.message(msg);
    }
    virtual void* get_mem_ptr(bh_base &base, bool copy2host, bool force_alloc, bool nullify) {
        return child.get_mem_ptr(base, copy2host, force_alloc, nullify);
    };
    virtual void set_mem_ptr(bh_base *base, bool host_ptr, void *mem) {
        return child.set_mem_ptr(base, host_ptr, mem);
    }
    virtual void* get_device_context() {
        return child.get_device_context();
    };
    virtual void set_device_context(void* device_context) {
        child.set_device_context(device_context);
    };
};

}} //namespace bohrium::component

#endif
