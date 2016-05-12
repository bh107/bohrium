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

#ifndef __BH_EXTMETHOD_HPP
#define __BH_EXTMETHOD_HPP

#include <string>
#include <cassert>
#include <exception>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include <bh_config_parser.hpp>
#include <bh_error.h>
#include <bh_ir.hpp>
#include <bh_opcode.h>

namespace bohrium {
namespace extmethod {

/* A extmethod in Bohrium is a function that implement a specific instruction.
 *
 * Requirements:
 *   - A extmethod must be compiled into a shared library e.g. .so, .dylib, or .dll.
 *     but multiple extmethods may be in the same shared library
 *
 *	 - A extmethod has to implement the ExtmethodImpl class; specifically,
 *	   the execute() method, which the parent component calls when it wants to
 *	   execute the instruction.
 *
 *   - A extmethod has to implement two C compatible functions, <name>_create() and
 *     <name>_destroy(), where <name> is the extmethod name.
 *
 *   - The parent search for the two C compatible functions in the shared libraries
 *     that the config file specifies under "libs".
 *
 *   - The ExtmethodFace class makes it easy for a parent to do this search
 *     and use the found execute() method.
 */

// Representation of a extmethod implementation, which is a virtual class
// that all extended methods should implement
class ExtmethodImpl {
  public:
    ExtmethodImpl() = default;
    virtual ~ExtmethodImpl() {};

    /* Execute an instruction
     *
     * @instr The extension method instruction to handle
     * @arg   Additional component specific argument
     * Throws exceptions on error
     */
    virtual void execute(bh_instruction *instr, void* arg) = 0;
};

// Representation of an extmethod interface, which consist of a create()
// and destroy() function.
class ExtmethodFace {
  private:
    // The name of the extmethod e.g. matmul or visualizer
    std::string _name;
    // Path to the shared library file e.g. .so, .dylib, or .dll
    std::string _lib_path;
    // Shared library handle
    void* _lib_handle;
    // Function pointer to the extmethod's create function
    ExtmethodImpl* (*_create)();
    // Function pointer to the extmethod's destroy function
    void (*_destroy)(ExtmethodImpl *extmethod);
    // Pointer to the implementation of the extended method
    ExtmethodImpl *_implementation;
  public:
    // Constructor that takes the config file of the parent component and
    // the name of the extmethod
    ExtmethodFace(const ConfigParser &parent_config, const std::string &name);
    ~ExtmethodFace();

    // Move Constructor
    ExtmethodFace(ExtmethodFace &&other) {
        if (this != &other)
        {
            _name = other._name;
            _lib_path = other._lib_path;
            _lib_handle = other._lib_handle;
            _create = other._create;
            _destroy = other._destroy;
            _implementation = other._implementation;
            other._name.clear();
            other._lib_path.clear();
            other._lib_handle = NULL;
            other._create = NULL;
            other._destroy = NULL;
            other._implementation = NULL;
        }
    }
    // No copy constructor use the move constructor instead.
    ExtmethodFace(const ExtmethodFace &other) = delete;

    // Get the extmethod implementation
    ExtmethodImpl* getImpl() { return _implementation; };

    /* Execute an instruction
     *
     * @instr The extension method instruction to handle
     * @arg   Additional component specific argument
     * Throws exceptions on error
     */
    void execute(bh_instruction *instr, void* arg) {
        assert(_implementation != NULL);
        return _implementation->execute(instr, arg);
    };
};

class ExtmethodNotFound : public std::exception
{
    std::string _msg;
public:
    ExtmethodNotFound(const std::string& msg) : _msg(msg) {}
    virtual const char* what() const throw() { return _msg.c_str(); }
};

}} //namespace bohrium::extmethod

#endif

