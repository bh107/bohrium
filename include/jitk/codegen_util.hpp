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

#include <bh_util.hpp>
#include <bh_type.hpp>
#include <bh_instruction.hpp>
#include <bh_extmethod.hpp>
#include <bh_config_parser.hpp>

#include <jitk/block.hpp>

#include <jitk/symbol_table.hpp>

#include <jitk/instruction.hpp>

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

inline std::vector<const bh_view*> scalar_replaced_input_only(const LoopB &block, const Scope *parent_scope, const std::set<bh_base*> &local_tmps) {
    std::vector<const bh_view*> result;

    const std::vector<InstrPtr> block_instr_list = block.getAllInstr();
    // We have to ignore output arrays and arrays that are accumulated
    std::set<bh_base *> ignore_bases;
    for (const InstrPtr &instr: block_instr_list) {
        if (not instr->operand.empty()) {
            ignore_bases.insert(instr->operand[0].base);
        }
        if (bh_opcode_is_accumulate(instr->opcode)) {
            ignore_bases.insert(instr->operand[1].base);
        }
    }
    // First we add a valid view to the set of 'candidates' and if we encounter the view again
    // we add it to the 'result'
    std::set<bh_view> candidates;
    for (const InstrPtr &instr: block_instr_list) {
        for(size_t i=1; i < instr->operand.size(); ++i) {
            const bh_view &input = instr->operand[i];
            if ((not bh_is_constant(&input)) and ignore_bases.find(input.base) == ignore_bases.end()) {
                if (local_tmps.find(input.base) == local_tmps.end() and
                    (parent_scope == nullptr or parent_scope->isArray(input))) {
                    if (util::exist(candidates, input)) { // 'input' is used multiple times
                        result.push_back(&input);
                    } else {
                        candidates.insert(input);
                    }
                }
            }
        }
    }

    return result;
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


} // jitk
} // bohrium
