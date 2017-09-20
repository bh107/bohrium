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
#include <jitk/fuser_cache.hpp>
#include <jitk/apply_fusion.hpp>


namespace bohrium {
namespace jitk {


// Write 'num' of spaces to 'out'
void spaces(std::stringstream &out, int num);

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

// Calculate the work group sizes.
// Return pair (global work size, local work size)
std::pair<uint32_t, uint32_t> work_ranges(uint64_t work_group_size, int64_t block_size);

// Write the kernel function arguments.
// The function 'type_writer' should write the backend specific data type names.
// The string 'array_type_prefix' specifies the type prefix for array pointers (e.g. "__global" in OpenCL)
void write_kernel_function_arguments(const SymbolTable &symbols,
                                     std::function<const char *(bh_type type)> type_writer,
                                     std::stringstream &ss, const char *array_type_prefix,
                                     const bool fortran_style_param);


// Writes a loop block, which corresponds to a parallel for-loop.
// The two functions 'type_writer' and 'head_writer' should write the
// backend specific data type names and for-loop headers respectively.
void write_loop_block(const SymbolTable &symbols,
                      const Scope *parent_scope,
                      const LoopB &block,
                      const ConfigParser &config,
                      const std::vector<const LoopB *> &threaded_blocks,
                      bool opencl,
                      bool fortran,
                      std::function<const char *(bh_type type)> type_writer,
                      std::function<void (const SymbolTable &symbols,
                                          Scope &scope,
                                          const LoopB &block,
                                          const ConfigParser &config,
                                          bool loop_is_peeled,
                                          const std::vector<const LoopB *> &threaded_blocks,
                                          std::stringstream &out)> head_writer,
                      std::stringstream &declares,
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

// Returns the blocks that can be parallelized in 'block' (incl. sub-blocks)
inline std::vector<const LoopB*> find_threaded_blocks(const Block &block, Statistics &stat,
                                                      uint64_t parallel_threshold) {
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

/* Handle execution of regular instructions
 * 'SelfType' most be a component implementation that exposes:
 *     - void write_kernel(...)
 * 'EngineType' most be a engine implementation that exposes:
 *     - set_constructor_flag(...)
 */
template<typename SelfType, typename EngineType>
void handle_cpu_execution(SelfType &self, bh_ir *bhir, EngineType &engine, const ConfigParser &config, Statistics &stat,
                          FuseCache &fcache) {
    using namespace std;

    const auto texecution = chrono::steady_clock::now();

    const bool strides_as_var = config.defaultGet<bool>("strides_as_var", true);
    const bool index_as_var = config.defaultGet<bool>("index_as_var", true);
    const bool const_as_var = config.defaultGet<bool>("const_as_var", true);
    const bool monolithic = config.defaultGet<bool>("monolithic", false);

    // Some statistics
    stat.record(bhir->instr_list);

    // Let's start by cleanup the instructions from the 'bhir'
    vector<bh_instruction*> instr_list;
    {
        set<bh_base*> syncs;
        set<bh_base*> frees;
        instr_list = remove_non_computed_system_instr(bhir->instr_list, syncs, frees);

        // Let's free device buffers and array memory
        for(bh_base *base: frees) {
            bh_data_free(base);
        }
    }

    // Set the constructor flag
    if (config.defaultGet<bool>("array_contraction", true)) {
        engine.set_constructor_flag(instr_list);
    } else {
        for (bh_instruction *instr: instr_list) {
            instr->constructor = false;
        }
    }

    // Let's get the block list
    const vector<Block> block_list = get_block_list(instr_list, config, fcache, stat, false);

    if (monolithic) {
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
        {
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
        }

        // Let's create the symbol table for the kernel
        const SymbolTable symbols(all_instr, all_non_temps, strides_as_var, index_as_var, const_as_var);
        stat.record(symbols);

        // Let's execute the kernel
        if (kernel_is_computing) { // We can skip this step if the kernel does no computation
            // Code generation
            stringstream ss;
            self.write_kernel(block_list, symbols, config, kernel_temps, ss);

            // Create the constant vector
            vector<const bh_instruction*> constants;
            constants.reserve(symbols.constIDs().size());
            for (const InstrPtr &instr: symbols.constIDs()) {
                constants.push_back(&(*instr));
            }

            // Let's execute the kernel
            engine.execute(ss.str(), symbols.getParams(), symbols.offsetStrideViews(), constants);
        }

        // Finally, let's cleanup
        for(bh_base *base: symbols.getFrees()) {
            bh_data_free(base);
        }
    } else {
        // When creating a regular kernels (a block-nest per shared library), we create one kernel at a time
        for(const Block &block: block_list) {
            assert(not block.isInstr());

            // Let's create the symbol table for the kernel
            const SymbolTable symbols(block.getAllInstr(), block.getLoop().getAllNonTemps(), strides_as_var,
                                      index_as_var, const_as_var);
            stat.record(symbols);

            // Let's execute the kernel
            if (not block.isSystemOnly()) { // We can skip this step if the kernel does no computation
                // Code generation
                stringstream ss;

                self.write_kernel({block}, symbols, config, {}, ss);

                // Create the constant vector
                vector<const bh_instruction*> constants;
                constants.reserve(symbols.constIDs().size());
                for (const InstrPtr &instr: symbols.constIDs()) {
                    constants.push_back(&(*instr));
                }

                // Let's execute the kernel
                engine.execute(ss.str(), symbols.getParams(), symbols.offsetStrideViews(), constants);
            }

            // Finally, let's cleanup
            for(bh_base *base: symbols.getFrees()) {
                bh_data_free(base);
            }
        }
    }
    stat.time_total_execution += chrono::steady_clock::now() - texecution;
}

/* Handle execution of regular instructions
 * 'SelfType' most be a component implementation that exposes:
 *     - void write_kernel(...)
 * 'EngineType' most be a engine implementation that exposes:
 *     - set_constructor_flag(...)
 *     - void copyToHost(...)
 *     - void copyToDevice(...)
 *     - void delBuffer(...)
 * 'child' can only be NULL when find_threaded_blocks() always returns one or more blocks
 */
template<typename SelfType, typename EngineType>
void handle_gpu_execution(SelfType &self, bh_ir *bhir, EngineType &engine, const ConfigParser &config, Statistics &stat,
                          FuseCache &fcache, component::ComponentFace *child) {
    using namespace std;

    const auto texecution = chrono::steady_clock::now();

    const bool verbose = config.defaultGet<bool>("verbose", false);
    const bool strides_as_var = config.defaultGet<bool>("strides_as_var", true);
    const bool index_as_var = config.defaultGet<bool>("index_as_var", true);
    const bool const_as_var = config.defaultGet<bool>("const_as_var", true);
    const uint64_t parallel_threshold = config.defaultGet<uint64_t>("parallel_threshold", 1000);

    // Some statistics
    stat.record(bhir->instr_list);

    // Let's start by cleanup the instructions from the 'bhir'
    vector<bh_instruction*> instr_list;
    {
        set<bh_base*> syncs;
        set<bh_base*> frees;
        instr_list = remove_non_computed_system_instr(bhir->instr_list, syncs, frees);

        // Let's copy sync'ed arrays back to the host
        engine.copyToHost(syncs);

        // Let's free device buffers and array memory
        for(bh_base *base: frees) {
            engine.delBuffer(base);
            bh_data_free(base);
        }
    }

    // Set the constructor flag
    if (config.defaultGet<bool>("array_contraction", true)) {
        engine.set_constructor_flag(instr_list);
    } else {
        for (bh_instruction *instr: instr_list) {
            instr->constructor = false;
        }
    }

    // Let's get the block list
    // NB: 'avoid_rank0_sweep' is set to true when we have a child to offload to.
    const vector<Block> block_list = get_block_list(instr_list, config, fcache, stat, child != NULL);

    for(const Block &block: block_list) {
        assert(not block.isInstr());

        // Let's create the symbol table for the kernel
        const SymbolTable symbols(block.getAllInstr(), block.getLoop().getAllNonTemps(), strides_as_var,
                                  index_as_var, const_as_var);
        stat.record(symbols);

        // We can skip a lot of steps if the kernel does no computation
        const bool kernel_is_computing = not block.isSystemOnly();

        // Find the parallel blocks
        const vector<const LoopB*> threaded_blocks = find_threaded_blocks(block, stat, parallel_threshold);

        // We might have to offload the execution to the CPU
        if (threaded_blocks.size() == 0 and kernel_is_computing) {
            if (verbose)
                cout << "Offloading to CPU\n";

            if (child == NULL) {
                throw runtime_error("handle_execution(): threaded_blocks cannot be empty when child == NULL!");
            }

            auto toffload = chrono::steady_clock::now();

            // Let's copy all non-temporary to the host
            engine.copyToHost(symbols.getParams());

            // Let's free device buffers
            for (bh_base *base: symbols.getFrees()) {
                engine.delBuffer(base);
            }

            // Let's send the kernel instructions to our child
            vector<bh_instruction> child_instr_list;
            for (const InstrPtr &instr: block.getAllInstr()) {
                child_instr_list.push_back(*instr);
            }
            bh_ir tmp_bhir(child_instr_list.size(), &child_instr_list[0]);
            child->execute(&tmp_bhir);
            stat.time_offload += chrono::steady_clock::now() - toffload;
            continue;
        }

        // Let's execute the kernel
        if (kernel_is_computing) {

            // We need a memory buffer on the device for each non-temporary array in the kernel
            engine.copyToDevice(symbols.getParams());

            // Code generation
            stringstream ss;
            self.write_kernel(block, symbols, config, threaded_blocks, ss);

            // Create the constant vector
            vector<const bh_instruction*> constants;
            constants.reserve(symbols.constIDs().size());
            for (const InstrPtr &instr: symbols.constIDs()) {
                constants.push_back(&(*instr));
            }

            // Let's execute the kernel
            engine.execute(ss.str(), symbols.getParams(), threaded_blocks, symbols.offsetStrideViews(), constants);
        }

        // Let's copy sync'ed arrays back to the host
        engine.copyToHost(symbols.getSyncs());

        // Let's free device buffers
        for(bh_base *base: symbols.getFrees()) {
            engine.delBuffer(base);
        }

        // Finally, let's cleanup
        for(bh_base *base: symbols.getFrees()) {
            bh_data_free(base);
        }
    }
    stat.time_total_execution += chrono::steady_clock::now() - texecution;
}

} // jitk
} // bohrium

#endif
