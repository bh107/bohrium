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
#pragma once

#include <string>

#include <bh_config_parser.hpp>
#include <bh_ir.hpp>
#include <bh_opcode.h>
#include <bh_extmethod.hpp>
#include <jitk/statistics.hpp>

namespace bohrium {
namespace component {

class ComponentImpl; // Forward declaration

/** A Component in Bohrium is a shared library that implement a specific functionality.
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

/** Representation of a component interface */
class ComponentFace {
private:
    // Shared library handle, which is null when the component is uninitiated
    void *_lib_handle;

    // Pointer to the implementation of the component, which is null when the component is uninitiated
    ComponentImpl *_implementation;

    // Function pointer to the component's create function
    ComponentImpl *(*_create)(unsigned int);

    // Function pointer to the component's destroy function
    void (*_destroy)(ComponentImpl *component);

public:
    /** Constructor that takes the path to the shared library and the stack level of the component to interface
     *
     * @param lib_path    The path to the shared library of the component to interface
     * @param stack_level The stack level of the component of the component to interface
     */
    ComponentFace(const std::string &lib_path, int stack_level);

    /** Default constructor which create an uninitiated component interface */
    ComponentFace() : _lib_handle(nullptr), _implementation(nullptr) {};

    // We only support the move assignment operator
    ComponentFace(const ComponentFace &other) = delete;

    ComponentFace(ComponentFace &&other) = delete;

    ~ComponentFace();

    /** Move constructor, which we need to make sure that `other` is left uninitiated */
    ComponentFace &operator=(ComponentFace &&other) noexcept {
        _lib_handle = other._lib_handle;
        _implementation = other._implementation;
        _create = other._create;
        _destroy = other._destroy;
        other._lib_handle = nullptr;
        other._implementation = nullptr;
        other._create = nullptr;
        other._destroy = nullptr;
        return *this;
    };

    bool initiated() const;

    void execute(BhIR *bhir);

    void extmethod(const std::string &name, bh_opcode opcode);

    std::string message(const std::string &msg);

    void *getMemoryPointer(bh_base &base, bool copy2host, bool force_alloc, bool nullify);

    void setMemoryPointer(bh_base *base, bool host_ptr, void *mem);

    void *getDeviceContext();

    void setDeviceContext(void *device_context);
};

// Representation of a component implementation, which is an abstract class
// that all Bohrium components should implement
class ComponentImpl {
protected:
    // Flag that indicate whether the component is enabled or disabled.
    // When disabled, the component should pass through instructions untouched to its child
    bool disabled = false;
public:

    // The level in the runtime stack starting a -1, which is the bridge,
    // 0 is the first component in the stack list, 1 is the second component etc.
    const int stack_level;
    // The configure file
    const ConfigParser config;

    // The interface of the child. Notice, the child might not exist i.e. `child.exist() == false`
    ComponentFace child;

    /** Constructor for a new component
     *
     * @param stack_level     The stack level of the new component
     * @param initiate_child  Flag: initiate the child (if any)
     */
    explicit ComponentImpl(int stack_level, bool initiate_child = true) : stack_level(stack_level),
                                                                          config(stack_level) {
        std::string child_lib_path = config.getChildLibraryPath();
        if (initiate_child and not child_lib_path.empty()) { // Has a child
            child = ComponentFace{ComponentImpl::config.getChildLibraryPath(), stack_level + 1};
        }
    }

    virtual ~ComponentImpl() = default; // NB: a destructor implementation must exist

    /** Execute a BhIR (graph of instructions)
     *
     * @bhir  The BhIR to execute
     * Throws exceptions on error
     */
    virtual void execute(BhIR *bhir) {
        child.execute(bhir);
    };

    /** Register a new extension method.
     *
     * @name   Name of the function
     * @opcode Opcode for the new function.
     * Throws exceptions on error
     */
    virtual void extmethod(const std::string &name, bh_opcode opcode) {
        child.extmethod(name, opcode);
    }

    /** Send and receive a message through the component stack
     *
     * @msg    The message to send
     * @return The received message
     * Throws exceptions on error
     */
    virtual std::string message(const std::string &msg) {
        return child.message(msg);
    }

    /** Get data pointer from the first VE in the runtime stack
     * NB: this doesn't include a flush
     *
     * @base         The base array that owns the data
     * @copy2host    Always copy the memory to main memory
     * @force_alloc  Force memory allocation
     * @nullify      Set the data pointer to NULL after returning
     * @return       The data pointer (NB: might point to device memory)
     * Throws exceptions on error
     */
    virtual void *getMemoryPointer(bh_base &base, bool copy2host, bool force_alloc, bool nullify) {
        return child.getMemoryPointer(base, copy2host, force_alloc, nullify);
    }

    /** Set data pointer in the first VE in the runtime stack
     * NB: The component will deallocate the memory when encountering a BH_FREE.
     *     Also, this doesn't include a flush
     *
     * @base      The base array that will own the data
     * @host_ptr  The pointer points to the host memory (main memory) as opposed to device memory
     * @mem       The data pointer
     * Throws exceptions on error
     */
    virtual void setMemoryPointer(bh_base *base, bool host_ptr, void *mem) {
        return child.setMemoryPointer(base, host_ptr, mem);
    }

    /** Get the device handle, such as OpenCL's cl_context, of the first VE in the runtime stack.
     * If the first VE isn't a device, NULL is returned.
     *
     * @return  The device handle
     * Throws exceptions on error
     */
    virtual void *getDeviceContext() {
        return child.getDeviceContext();
    }

    /** Set the device context, such as CUDA's context, of the first VE in the runtime stack.
     * If the first VE isn't a device, nothing happens
     *
     * @device_context  The new device context
     * Throws exceptions on error
     */
    virtual void setDeviceContext(void *device_context) {
        child.setDeviceContext(device_context);
    }
};

// Representation of a vector engine component
// All vector engines should inherit from this class
class ComponentVE : public ComponentImpl {
public:
    // Known extension methods
    std::map<bh_opcode, extmethod::ExtmethodFace> extmethods;

    // The child's known extension methods
    std::set<bh_opcode> child_extmethods;

    // Trivial constructor
    ComponentVE(int stack_level) : ComponentImpl(stack_level) {}
};

}
} //namespace bohrium::component
