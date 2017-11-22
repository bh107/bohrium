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
#include <bh_component.hpp>
#include <bh_extmethod.hpp>
#include <bh_config_parser.hpp>
#include <jitk/block.hpp>
#include <jitk/base_db.hpp>
#include <jitk/instruction.hpp>
#include <jitk/view.hpp>
#include <jitk/fuser_cache.hpp>
#include <jitk/codegen_cache.hpp>
#include <jitk/apply_fusion.hpp>
#include <jitk/statistics.hpp>

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
std::string hash_filename(size_t compilation_hash, size_t source_hash, std::string file_extension);

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

// Calculate the work group sizes.
// Return pair (global work size, local work size)
std::pair<uint32_t, uint32_t> work_ranges(uint64_t work_group_size, int64_t block_size);

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

// Returns the blocks that can be parallelized in 'block' (incl. sub-blocks)
inline std::vector<const LoopB*> find_threaded_blocks(const Block &block, Statistics &stat, uint64_t parallel_threshold) {
    std::vector<const LoopB*> threaded_blocks;
    uint64_t total_threading;
    tie(threaded_blocks, total_threading) = util_find_threaded_blocks(block.getLoop());
    if (total_threading < parallel_threshold) {
        for (const InstrPtr instr: block.getAllInstr()) {
            if (not bh_opcode_is_system(instr->opcode)) {
                stat.threading_below_threshold += bh_nelements(instr->operand[0]);
            }
        }
    }
    return threaded_blocks;
}

template<typename SelfType>
void create_monolithic_kernel(SelfType &self, std::map<std::string, bool> kernel_config, const std::vector<Block> &block_list) {
    using namespace std;

    // When creating a monolithic kernel (all instructions in one shared library), we first combine
    // the instructions and non-temps from each different block and then we create the kernel
    vector<InstrPtr> all_instr;
    set<bh_base *> all_non_temps;
    bool kernel_is_computing = false;
    for(const Block &block: block_list) {
        assert(not block.isInstr());
        block.getAllInstr(all_instr);
        block.getLoop().getAllNonTemps(all_non_temps);
        if (not block.isSystemOnly()) {
            kernel_is_computing = true;
        }
    }
    // Let's find the arrays that are allocated and freed between blocks in the kernel
    vector<bh_base*> kernel_temps;
    set<bh_base*> constructors;
    for(const InstrPtr &instr: all_instr) {
        if (instr->constructor) {
            assert(instr->operand[0].base != NULL);
            constructors.insert(instr->operand[0].base);
        } else if (instr->opcode == BH_FREE and util::exist(constructors, instr->operand[0].base)) {
            kernel_temps.push_back(instr->operand[0].base);
            all_non_temps.erase(instr->operand[0].base);
        }
    }

    // Let's create the symbol table for the kernel
    const SymbolTable symbols(all_instr, all_non_temps, kernel_config);
    self.stat.record(symbols);

    // Let's execute the kernel
    if (kernel_is_computing) { // We can skip this step if the kernel does no computation
        // Create the constant vector
        vector<const bh_instruction*> constants;
        constants.reserve(symbols.constIDs().size());
        for (const InstrPtr &instr: symbols.constIDs()) {
            constants.push_back(&(*instr));
        }

        const auto lookup = self.codegen_cache.get(block_list, symbols);
        if(lookup.second) {
            // In debug mode, we check that the cached source code is correct
            #ifndef NDEBUG
                stringstream ss;
                self.write_kernel(block_list, symbols, kernel_temps, ss);
                if (ss.str().compare(lookup.first) != 0) {
                    cout << "\nCached source code: \n" << lookup.first;
                    cout << "\nReal source code: \n" << ss.str();
                    assert(1 == 2);
                }
            #endif
            self.execute(lookup.first, symbols.getParams(), symbols.offsetStrideViews(), constants);
        } else {
            const auto tcodegen = chrono::steady_clock::now();
            stringstream ss;
            self.write_kernel(block_list, symbols, kernel_temps, ss);
            string source = ss.str();
            self.stat.time_codegen += chrono::steady_clock::now() - tcodegen;
            self.execute(source, symbols.getParams(), symbols.offsetStrideViews(), constants);
            self.codegen_cache.insert(std::move(source), block_list, symbols);
        }
    }

    // Finally, let's cleanup
    for(bh_base *base: symbols.getFrees()) {
        bh_data_free(base);
    }
}

template<typename SelfType>
void create_kernel(SelfType &self, std::map<std::string, bool> kernel_config, const std::vector<Block> &block_list) {
    using namespace std;

    // When creating a regular kernels (a block-nest per shared library), we create one kernel at a time
    for(const Block &block: block_list) {
        assert(not block.isInstr());

        // Let's create the symbol table for the kernel
        const SymbolTable symbols(block.getAllInstr(), block.getLoop().getAllNonTemps(), kernel_config);
        self.stat.record(symbols);

        // Let's execute the kernel
        if (not block.isSystemOnly()) { // We can skip this step if the kernel does no computation

            // Create the constant vector
            vector<const bh_instruction*> constants;
            constants.reserve(symbols.constIDs().size());
            for (const InstrPtr &instr: symbols.constIDs()) {
                constants.push_back(&(*instr));
            }
            const auto lookup = self.codegen_cache.get({ block }, symbols);
            if(lookup.second) {
                // In debug mode, we check that the cached source code is correct
                #ifndef NDEBUG
                    stringstream ss;
                    self.write_kernel({block}, symbols, {}, ss);
                    if (ss.str().compare(lookup.first) != 0) {
                        cout << "\nCached source code: \n" << lookup.first;
                        cout << "\nReal source code: \n" << ss.str();
                        assert(1 == 2);
                    }
                #endif
                self.execute(lookup.first, symbols.getParams(), symbols.offsetStrideViews(), constants);
            } else {
                const auto tcodegen = chrono::steady_clock::now();
                stringstream ss;
                self.write_kernel({block}, symbols, {}, ss);
                string source = ss.str();
                self.stat.time_codegen += chrono::steady_clock::now() - tcodegen;
                self.execute(source, symbols.getParams(), symbols.offsetStrideViews(), constants);
                self.codegen_cache.insert(std::move(source), {block}, symbols);
            }
        }

        // Finally, let's cleanup
        for(bh_base *base: symbols.getFrees()) {
            bh_data_free(base);
        }
    }
}

} // jitk
} // bohrium
