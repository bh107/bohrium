/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium team:
http://bohrium.bitbucket.org

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
#include <iostream>
#include <sstream>

#include "BhInstruction.hpp"
#include <bh_component.hpp>

namespace bhxx {

/**
 *  Encapsulation of communication with Bohrium runtime.
 *  Implemented as a Singleton.
 *
 *  \note  Not thread-safe.
 */
class Runtime {
  public:
    Runtime();

    // Get the singleton instance of the Runtime class
    static Runtime& instance() {
        static Runtime instance;
        return instance;
    }

    // Create and enqueue a new bh_instruction based on `opcode` and a variadic
    // pack of BhArrays and at most one scalar value
    template <typename T, typename... Ts>
    void enqueue(bh_opcode opcode, T& op, Ts&... ops);

    /** Enqueue any BhInstruction object */
    void enqueue(BhInstruction instr);

    // We have to handle random specially because of the `BH_R123` scalar type
    void enqueue_random(BhArray<uint64_t>& out, uint64_t seed, uint64_t key);

    // Enqueue an extension method
    template <typename T>
    void enqueue_extmethod(const std::string& name, BhArray<T>& out, BhArray<T>& in1,
                           BhArray<T>& in2);

    /** Schedule a base object for deletion
     *
     * Will call BH_FREE on it first at the next flush
     */
    void enqueue_deletion(std::unique_ptr<BhBase> base_ptr);

    // Send enqueued instructions to Bohrium for execution
    void flush();

    // Send and receive a message through the component stack
    std::string message(const std::string &msg);

    ~Runtime() { flush(); }

    Runtime(Runtime&&) = default;
    Runtime& operator=(Runtime&&) = default;
    Runtime(const Runtime&)       = delete;
    Runtime& operator=(const Runtime&) = delete;

  private:
    //@{
    /** BH_FREE for arrays is special, since we deal with the deletion of the
     * base implictly via the BhBaseDeleter (which in turn calls
     * enqueue_deletion in this object).
     *
     * This function just resets the shared pointers of the array, which
     * might trigger the call of enqueue_deletion, but only if the
     * array is really no longer needed.
     * */
    template <typename T>
    void bh_free(BhArray<T>& ary);
    //@}

    // The lazy evaluated instructions
    std::vector<bh_instruction> instr_list;

    // Unique pointers to base objects, which are to be
    // purged after the next flush
    std::vector<std::unique_ptr<BhBase>> bases_for_deletion;

    // Bohrium Configuration
    bohrium::ConfigParser config;

    // The Bohrium Runtime i.e. the child of this component
    bohrium::component::ComponentFace runtime;

    // Mapping an extension method name to an opcode id
    std::map<std::string, bh_opcode> extmethods;

    // The opcode id for the next new extension method
    bh_opcode extmethod_next_opcode_id;
};

//
// ----------------------------------------------------------
//

template <typename T, typename... Ts>
void Runtime::enqueue(bh_opcode opcode, T& op, Ts&... ops) {
    if (opcode == BH_FREE) {
        // BH_FREE is special, see the bh_free function why.
        assert(sizeof...(Ts) == 0);
        bh_free(op);
    } else {
        BhInstruction instr(opcode);
        instr.append_operand(op, ops...);
        enqueue(std::move(instr));
    }
}

template <typename T>
void Runtime::enqueue_extmethod(const std::string& name, BhArray<T>& out, BhArray<T>& in1,
                                BhArray<T>& in2) {
    bh_opcode opcode;

    // Look for the extension opcode
    auto it = extmethods.find(name);
    if (it != extmethods.end()) {
        opcode = it->second;
    } else {
        // Add it and tell rest of Bohrium about this new extmethod
        opcode = extmethod_next_opcode_id++;
        runtime.extmethod(name.c_str(), opcode);
        extmethods.insert(std::pair<std::string, bh_opcode>(name, opcode));
    }

    // Now that we have an opcode, let's enqueue the instruction
    enqueue(opcode, out, in1, in2);
}

template <typename T>
void Runtime::bh_free(BhArray<T>& ary) {
    // Calling BH_FREE on an array with external
    // storage management is undefined behaviour
    if (!ary.base->own_memory()) {
        throw std::runtime_error(
              "Cannot call BH_FREE on a BhArray object, which uses external storage "
              "in its BhBase.");
    }

    // BH_FREE is special because it is automatically invoked
    // by the deleter of the shared pointer to BhBase if the last
    // array referencing BhBase is deleted.
    // So instead of actually deleting anything we will just
    // remove our reference to the BhBase instead
    ary.base.reset();
}
}
