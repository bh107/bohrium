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

#ifndef __BH_JITK_CODEGEN_UTIL_H
#define __BH_JITK_CODEGEN_UTIL_H

#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <functional>

#include <bh_util.hpp>
#include <bh_type.h>
#include <bh_instruction.hpp>
#include <bh_component.hpp>
#include <bh_extmethod.hpp>
#include <jitk/block.hpp>
#include <jitk/base_db.hpp>
#include <jitk/kernel.hpp>
#include <bh_config_parser.hpp>

namespace bohrium {
namespace jitk {


// Write 'num' of spaces to 'out'
void spaces(std::stringstream &out, int num);


// Write the kernel function arguments.
// The function 'type_writer' should write the backend specific data type names.
// The string 'array_type_prefix' specifies the type prefix for array pointers (e.g. "__global" in OpenCL)
void write_kernel_function_arguments(const Kernel &kernel, const SymbolTable &symbols,
                                     const std::vector<const bh_view*> &offset_strides,
                                     std::function<const char *(bh_type type)> type_writer,
                                     std::stringstream &ss, const char *array_type_prefix = NULL);


// Writes a loop block, which corresponds to a parallel for-loop.
// The two functions 'type_writer' and 'head_writer' should write the
// backend specific data type names and for-loop headers respectively.
void write_loop_block(const SymbolTable &symbols,
                      const Scope *parent_scope,
                      const LoopB &block,
                      const ConfigParser &config,
                      const std::vector<const LoopB *> &threaded_blocks,
                      bool opencl,
                      std::function<const char *(bh_type type)> type_writer,
                      std::function<void (const SymbolTable &symbols,
                                          Scope &scope,
                                          const LoopB &block,
                                          const ConfigParser &config,
                                          bool loop_is_peeled,
                                          const std::vector<const LoopB *> &threaded_blocks,
                                          std::stringstream &out)> head_writer,
                      std::stringstream &out);

// Sets the constructor flag of each instruction in 'instr_list'
// 'remotely_allocated_bases' is a collection of array bases already remotely allocated
template<typename T>
void util_set_constructor_flag(std::vector<bh_instruction *> &instr_list, const T &remotely_allocated_bases) {
    std::set<bh_base*> initiated; // Arrays initiated in 'instr_list'
    for(bh_instruction *instr: instr_list) {
        instr->constructor = false;
        for (size_t o = 0; o < instr->operand.size(); ++o) {
            const bh_view &v = instr->operand[o];
            if (not bh_is_constant(&v)) {
                assert(v.base != NULL);
                if (v.base->data == NULL and not (util::exist_nconst(initiated, v.base)
                                                  or util::exist_nconst(remotely_allocated_bases, v.base))) {
                    if (o == 0) { // It is only the output that is initiated
                        initiated.insert(v.base);
                        instr->constructor = true;
                    }
                }
            }
        }
    }
}

// Handle the extension methods within the 'bhir'
void util_handle_extmethod(component::ComponentImpl *self,
                           bh_ir *bhir,
                           std::map<bh_opcode, extmethod::ExtmethodFace> &extmethods);

// Handle the extension methods within the 'bhir'
// This version takes a child component and possible an engine that must have a copyToHost() method
template<typename T>
void util_handle_extmethod(component::ComponentImpl *self,
                           bh_ir *bhir,
                           std::map<bh_opcode, extmethod::ExtmethodFace> &extmethods,
                           std::set<bh_opcode> &child_extmethods,
                           component::ComponentFace &child,
                           T *acc_engine = NULL) {

    std::vector<bh_instruction> instr_list;
    for (bh_instruction &instr: bhir->instr_list) {
        auto ext = extmethods.find(instr.opcode);
        auto childext = child_extmethods.find(instr.opcode);

        if (ext != extmethods.end() or childext != child_extmethods.end()) {
            // Execute the instructions up until now
            bh_ir b;
            b.instr_list = instr_list;
            self->execute(&b);
            instr_list.clear();

            if (ext != extmethods.end()) {
                // Execute the extension method
                ext->second.execute(&instr, acc_engine);
            } else if (childext != child_extmethods.end()) {
                // We let the child component execute the instruction
                std::set<bh_base *> ext_bases = instr.get_bases();
                if (acc_engine != NULL) {
                    acc_engine->copyToHost(ext_bases);
                }
                std::vector<bh_instruction> child_instr_list;
                child_instr_list.push_back(instr);
                b.instr_list = child_instr_list;
                child.execute(&b);
            }
        } else {
            instr_list.push_back(instr);
        }
    }
    bhir->instr_list = instr_list;
}

} // jitk
} // bohrium

#endif
