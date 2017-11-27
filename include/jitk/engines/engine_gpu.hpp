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

#include "engine.hpp"

#include <bh_config_parser.hpp>
#include <jitk/statistics.hpp>
#include <jitk/compiler.hpp>

#include <bh_view.hpp>
#include <bh_component.hpp>
#include <bh_instruction.hpp>

namespace bohrium {
namespace jitk {

class EngineGPU : public Engine {
public:
    // OpenCL compile flags
    const std::string compile_flg;
    // Default device type
    const std::string default_device_type;
    // Default platform number
    const int platform_no;
    // Record profiling statistics
    const bool prof;

    EngineGPU(const ConfigParser &config, Statistics &stat) :
      Engine(config, stat),
      compile_flg(jitk::expand_compile_cmd(config.defaultGet<std::string>("compiler_flg", ""), "", "", config.file_dir.string())),
      default_device_type(config.defaultGet<std::string>("device_type", "auto")),
      platform_no(config.defaultGet<int>("platform_no", -1)),
      prof(config.defaultGet<bool>("prof", false)) {
    }

    virtual ~EngineGPU() {}

    virtual void copyToHost(const std::set<bh_base*> &bases) = 0;
    virtual void allBasesToHost() = 0;
    virtual void copyToDevice(const std::set<bh_base*> &base_list) = 0;
    virtual void delBuffer(bh_base* &base) = 0;
    virtual void write_kernel(const Block &block,
                              const SymbolTable &symbols,
                              const std::vector<const LoopB*> &threaded_blocks,
                              std::stringstream &ss) = 0;
    virtual void execute(const std::string &source,
                         const std::vector<bh_base*> &non_temps,
                         const std::vector<const LoopB*> &threaded_blocks,
                         const std::vector<const bh_view*> &offset_strides,
                         const std::vector<const bh_instruction*> &constants) = 0;

    void handle_execution(component::ComponentImplWithChild &comp, BhIR *bhir) {
        using namespace std;

        const auto texecution = chrono::steady_clock::now();

        map<string, bool> kernel_config = {
            { "strides_as_var", config.defaultGet<bool>("strides_as_var", true) },
            { "index_as_var",   config.defaultGet<bool>("index_as_var",   true) },
            { "const_as_var",   config.defaultGet<bool>("const_as_var",   true) },
            { "use_volatile",   config.defaultGet<bool>("use_volatile",  false) }
        };
        const uint64_t parallel_threshold = config.defaultGet<uint64_t>("parallel_threshold", 1000);

        // Some statistics
        stat.record(*bhir);

        // Let's start by cleanup the instructions from the 'bhir'
        vector<bh_instruction*> instr_list;

        set<bh_base*> frees;
        instr_list = jitk::remove_non_computed_system_instr(bhir->instr_list, frees);

        // Let's free device buffers and array memory
        for(bh_base *base: frees) {
            delBuffer(base);
            bh_data_free(base);
        }

        // Set the constructor flag
        if (config.defaultGet<bool>("array_contraction", true)) {
            set_constructor_flag(instr_list);
        } else {
            for (bh_instruction *instr: instr_list) {
                instr->constructor = false;
            }
        }

        // Let's get the block list
        // NB: 'avoid_rank0_sweep' is set to true when we have a child to offload to.
        const vector<jitk::Block> block_list = get_block_list(instr_list, config, fcache, stat, &(comp.child) != nullptr);

        for(const jitk::Block &block: block_list) {
            assert(not block.isInstr());

            // Let's create the symbol table for the kernel
            const jitk::SymbolTable symbols(
              block.getAllInstr(),
              block.getLoop().getAllNonTemps(),
              kernel_config["use_volatile"],
              kernel_config["strides_as_var"],
              kernel_config["index_as_var"],
              kernel_config["const_as_var"]
            );
            stat.record(symbols);

            // We can skip a lot of steps if the kernel does no computation
            const bool kernel_is_computing = not block.isSystemOnly();

            // Find the parallel blocks
            const vector<const jitk::LoopB*> threaded_blocks = find_threaded_blocks(block, stat, parallel_threshold);

            // We might have to offload the execution to the CPU
            if (threaded_blocks.size() == 0 and kernel_is_computing) {
                if (config.defaultGet<bool>("verbose", false)) {
                    cout << "Offloading to CPU\n";
                }

                if (&(comp.child) == nullptr) {
                    throw runtime_error("handle_execution(): threaded_blocks cannot be empty when child == NULL!");
                }

                auto toffload = chrono::steady_clock::now();

                // Let's copy all non-temporary to the host
                const vector<bh_base*> v = symbols.getParams();
                copyToHost(set<bh_base*>(v.begin(), v.end()));

                // Let's free device buffers
                for (bh_base *base: symbols.getFrees()) {
                    delBuffer(base);
                }

                // Let's send the kernel instructions to our child
                vector<bh_instruction> child_instr_list;
                for (const jitk::InstrPtr &instr: block.getAllInstr()) {
                    child_instr_list.push_back(*instr);
                }
                BhIR tmp_bhir(std::move(child_instr_list), bhir->getSyncs());
                comp.child.execute(&tmp_bhir);
                stat.time_offload += chrono::steady_clock::now() - toffload;
                continue;
            }

            // Let's execute the kernel
            if (kernel_is_computing) {
                // We need a memory buffer on the device for each non-temporary array in the kernel
                const vector<bh_base*> v = symbols.getParams();
                copyToDevice(set<bh_base*>(v.begin(), v.end()));

                // Create the constant vector
                vector<const bh_instruction*> constants;
                constants.reserve(symbols.constIDs().size());
                for (const jitk::InstrPtr &instr: symbols.constIDs()) {
                    constants.push_back(&(*instr));
                }

                const auto lookup = codegen_cache.get({ block }, symbols);
                if(lookup.second) {
                    // In debug mode, we check that the cached source code is correct
                    #ifndef NDEBUG
                        stringstream ss;
                        write_kernel(block, symbols, threaded_blocks, ss);
                        if (ss.str().compare(lookup.first) != 0) {
                            cout << "\nCached source code: \n" << lookup.first;
                            cout << "\nReal source code: \n" << ss.str();
                            assert(1 == 2);
                        }
                    #endif
                    execute(lookup.first, symbols.getParams(), threaded_blocks, symbols.offsetStrideViews(), constants);
                } else {
                    const auto tcodegen = chrono::steady_clock::now();
                    stringstream ss;
                    write_kernel(block, symbols, threaded_blocks, ss);
                    string source = ss.str();
                    stat.time_codegen += chrono::steady_clock::now() - tcodegen;
                    execute(source, symbols.getParams(), threaded_blocks, symbols.offsetStrideViews(), constants);
                    codegen_cache.insert(std::move(source), { block }, symbols);
                }
            }

            // Let's copy sync'ed arrays back to the host
            copyToHost(bhir->getSyncs());

            // Let's free device buffers
            for(bh_base *base: symbols.getFrees()) {
                delBuffer(base);
            }

            // Finally, let's cleanup
            for(bh_base *base: symbols.getFrees()) {
                bh_data_free(base);
            }
        }
        stat.time_total_execution += chrono::steady_clock::now() - texecution;
    }

    template <typename T>
    void handle_extmethod(T &comp, BhIR *bhir, std::set<bh_opcode> child_extmethods) {
        std::vector<bh_instruction> instr_list;

        for (bh_instruction &instr: bhir->instr_list) {
            auto ext = comp.extmethods.find(instr.opcode);
            auto childext = child_extmethods.find(instr.opcode);

            if (ext != comp.extmethods.end() or childext != child_extmethods.end()) {
                // Execute the instructions up until now
                BhIR b(std::move(instr_list), bhir->getSyncs());
                comp.execute(&b);
                instr_list.clear(); // Notice, it is legal to clear a moved vector.

                if (ext != comp.extmethods.end()) {
                    const auto texecution = std::chrono::steady_clock::now();
                    ext->second.execute(&instr, &*this); // Execute the extension method
                    stat.time_ext_method += std::chrono::steady_clock::now() - texecution;
                } else if (childext != child_extmethods.end()) {
                    // We let the child component execute the instruction
                    std::set<bh_base *> ext_bases = instr.get_bases();

                    copyToHost(ext_bases);

                    std::vector<bh_instruction> child_instr_list;
                    child_instr_list.push_back(instr);
                    b.instr_list = child_instr_list;
                    comp.child.execute(&b);
                }
            } else {
                instr_list.push_back(instr);
            }
        }

        bhir->instr_list = instr_list;
    }
};

}} // namespace
