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
#ifndef __BOHRIUM_BRIDGE_BHXX_RUNTIME
#define __BOHRIUM_BRIDGE_BHXX_RUNTIME
#include <iostream>
#include <sstream>

#include <bh_component.hpp>
#include <bh_instruction.hpp>
#include "multi_array.hpp"

#include "multi_array.hpp"

namespace bhxx {


/**
 *  Encapsulation of communication with Bohrium runtime.
 *  Implemented as a Singleton.
 *
 *  Note: Not thread-safe.
 */
class Runtime {
private:
    std::vector<bh_instruction> instr_list;      // The lazy evaluated instructions
    std::vector<bh_base*> free_list;             // The bases that has to freed at the next flush

    bohrium::ConfigParser config;                // Bohrium Configuration
    bohrium::component::ComponentFace runtime;   // The Bohrium Runtime i.e. the child of this component

    std::map<std::string, bh_opcode> extmethods; // Mapping an extension method name to an opcode id
    size_t extmethod_next_opcode_id;             // The opcode id for the next new extension method

public:
    Runtime() : config(-1), // stack level -1 is the bridge
                runtime(config.getChildLibraryPath(), 0), // and child is stack level 0
                extmethod_next_opcode_id(BH_MAX_OPCODE_ID+1) {}

    // Get the singleton instance of the Runtime class
    static Runtime& instance() {
        static Runtime instance;
        return instance;
    };

    // `instr_append_operand()` append an operand to the list of instruction in `instr`
    // Variadic base case: appending one array
    template <typename T>
    void instr_append_operand(bh_instruction &instr, BhArray<T> ary) {
        bh_view view;
        view.base = ary.base->base;
        view.start = ary.offset;
        view.ndim = ary.shape.size();
        std::copy(ary.shape.begin(), ary.shape.end(), &view.shape[0]);
        std::copy(ary.stride.begin(), ary.stride.end(), &view.stride[0]);
        instr.operand.push_back(view);
    }
    // Variadic base case: appending one scalar
    template <typename T>
    void instr_append_operand(bh_instruction &instr, T scalar) {
        bh_view view;
        view.base = nullptr;
        instr.operand.push_back(view);
        bxx::assign_const_type(&instr.constant, scalar);
    }
    // Variadic case: appending one array and continue
    template <typename T, typename... Ts>
    void instr_append_operand(bh_instruction &instr, BhArray<T> ary, Ts... ops) {
        instr_append_operand(instr, ary);
        instr_append_operand(instr, ops...);
    }
    // Variadic case: appending one scalar and continue
    template <typename T, typename... Ts>
    void instr_append_operand(bh_instruction &instr, T scalar, Ts... ops) {
        instr_append_operand(instr, scalar);
        instr_append_operand(instr, ops...);
    }
    // Create and enqueue a new bh_instruction based on `opcode` and a variadic pack of BhArrays
    // and at most one scalar value
    template <typename... Ts>
    void enqueue(bh_opcode opcode, Ts... ops) {
        bh_instruction instr;
        instr.opcode = opcode;
        instr_append_operand(instr, ops...);
        instr_list.push_back(instr);
    }

    // We have to handle random specially because of the `BH_R123` scalar type
    void enqueue_random(BhArray<uint64_t> &out, uint64_t seed, uint64_t key) {
        bh_instruction instr;
        instr.opcode = BH_RANDOM;
        // Append the output array
        instr_append_operand(instr, out);

        // Append the special BH_R123 constant
        bh_view view;
        view.base = nullptr;
        instr.operand.push_back(view);
        instr.constant.type = BH_R123;
        instr.constant.value.r123.start = seed;
        instr.constant.value.r123.key   = key;
        instr_list.push_back(instr);
    }

    // We have to handle free specially because it takes a `BhBase` and must maintain the `free_list`
    template<typename T>
    void enqueue_free(BhBase<T> &base) {
        bh_view view;
        view.base = base.base;
        view.start = 0;
        view.ndim = 1;
        view.shape[0] = base.base->nelem;
        view.stride[0] = 1;
        bh_instruction instr;
        instr.opcode = BH_FREE;
        instr.operand.push_back(view);
        instr_list.push_back(instr);
        free_list.push_back(base.base);
    }

    // Enqueue an extension method
    template<typename T>
    void enqueue_extmethod(const std::string& name, BhArray<T> &out, BhArray<T> &in1, BhArray<T> &in2) {
        bh_opcode opcode;

        // Look for the extension opcode
        auto it = extmethods.find(name);
        if (it != extmethods.end()) {   // Got it
            opcode = it->second;
        } else {                        // Add it
            opcode = extmethod_next_opcode_id++;
            runtime.extmethod(name.c_str(), opcode); // Tell the rest of Bohrium about this new extmethod
            extmethods.insert(std::pair<std::string, bh_opcode>(name, opcode));
        }

        // Now that we have an opcode, let's enqueue the instruction
        enqueue(opcode, out, in1, in2);
    }

    // Send enqueued instructions to Bohrium for execution
    void flush() {
        bh_ir bhir = bh_ir(instr_list.size(), &instr_list[0]);
        runtime.execute(&bhir);
        instr_list.clear();
        for(bh_base *base: free_list) {
            delete base;
        }
        free_list.clear();
    }

    ~Runtime() {
        flush();
    }
};

}
#endif
