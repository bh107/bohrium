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

#include <bh_type.h>
#include <jitk/block.hpp>
#include <jitk/base_db.hpp>
#include <jitk/kernel.hpp>
#include <bh_config_parser.hpp>

namespace bohrium {
namespace jitk {


// Write 'num' of spaces to 'out'
void spaces(std::stringstream &out, int num) {
    for (int i = 0; i < num; ++i) {
        out << " ";
    }
}


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

} // jitk
} // bohrium

#endif
