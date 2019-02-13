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

#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <functional>
#include <boost/filesystem/path.hpp>

#include <bohrium/bh_util.hpp>
#include <bohrium/bh_type.hpp>
#include <bohrium/bh_instruction.hpp>
#include <bohrium/bh_extmethod.hpp>
#include <bohrium/bh_config_parser.hpp>

#include <bohrium/jitk/block.hpp>
#include <bohrium/jitk/symbol_table.hpp>
#include <bohrium/jitk/instruction.hpp>
#include <bohrium/jitk/iterator.hpp>

namespace bohrium {
namespace jitk {

// Does 'instr' reduce over the innermost axis?
// Notice, that such a reduction computes each output element completely before moving
// to the next element.
inline bool sweeping_innermost_axis(InstrPtr instr) {
    if (not bh_opcode_is_sweep(instr->opcode)) {
        return false;
    }
    assert(instr->operand.size() == 3);
    return instr->sweep_axis() == instr->operand[1].ndim - 1;
}

// Returns the filename of a the given hashes and file extension
// compilation_hash is the hash of the compile command and source_hash is the hash of the source code
std::string hash_filename(uint64_t compilation_hash, size_t source_hash, std::string file_extension);

// Write `src` to file in `dir` using `filename`
boost::filesystem::path write_source2file(const std::string &src,
                                          const boost::filesystem::path &dir,
                                          const std::string &filename,
                                          bool verbose);

// Returns the path to the tmp dir
boost::filesystem::path get_tmp_path(const ConfigParser &config);

// Tries five times to create directories recursively
// Useful when multiple processes runs on the same filesystem
void create_directories(const boost::filesystem::path &path);

// Order all sweep instructions by the viewID of their first operand.
// This makes the source of the kernels more identical, which improve the code and compile caches.
std::vector<InstrPtr> order_sweep_set(const std::set<InstrPtr> &sweep_set, const SymbolTable &symbols);

// Returns True when `view` is accessing row major style
bool row_major_access(const bh_view &view);

// Returns True when all views in `instr` is accessing row major style
bool row_major_access(const bh_instruction &instr);

// Transpose the instructions in `instr_list` to make them access column major style
void to_column_major(std::vector<bh_instruction> &instr_list);

} // jitk
} // bohrium
