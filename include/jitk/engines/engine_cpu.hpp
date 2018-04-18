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
#include <jitk/apply_fusion.hpp>

#include <bh_view.hpp>
#include <bh_component.hpp>
#include <bh_instruction.hpp>
#include <bh_main_memory.hpp>

namespace bohrium {
namespace jitk {

class EngineCPU : public Engine {
public:
    EngineCPU(const ConfigParser &config, Statistics &stat) : Engine(config, stat) {}

    virtual ~EngineCPU() {}

    virtual void writeKernel(const std::vector<Block> &block_list,
                             const SymbolTable &symbols,
                             const std::vector<bh_base *> &kernel_temps,
                             uint64_t codegen_hash,
                             std::stringstream &ss) = 0;

    virtual void execute(const jitk::SymbolTable &symbols,
                         const std::string &source,
                         uint64_t codegen_hash,
                         const std::vector<const bh_instruction *> &constants) = 0;

    virtual void handleExecution(BhIR *bhir) {
        using namespace std;

        const auto texecution = chrono::steady_clock::now();

        map<string, bool> kernel_config = {
                {"strides_as_var", config.defaultGet<bool>("strides_as_var", true)},
                {"index_as_var",   config.defaultGet<bool>("index_as_var", true)},
                {"const_as_var",   config.defaultGet<bool>("const_as_var", true)},
                {"use_volatile",   config.defaultGet<bool>("use_volatile", false)}
        };

        // Some statistics
        stat.record(*bhir);

        // Let's start by cleanup the instructions from the 'bhir'
        set<bh_base *> frees;
        vector<bh_instruction *> instr_list = jitk::remove_non_computed_system_instr(bhir->instr_list, frees);

        // Let's free device buffers and array memory
        for (bh_base *base: frees) {
            bh_data_free(base);
        }

        // Set the constructor flag
        if (config.defaultGet<bool>("array_contraction", true)) {
            setConstructorFlag(instr_list);
        } else {
            for (bh_instruction *instr: instr_list) {
                instr->constructor = false;
            }
        }

        // Let's get the block list
        const vector<jitk::Block> block_list = get_block_list(instr_list, config, fcache, stat, false);

        if (config.defaultGet<bool>("monolithic", false)) {
            createMonolithicKernel(kernel_config, block_list);
        } else {
            createKernel(kernel_config, block_list);
        }
        stat.time_total_execution += chrono::steady_clock::now() - texecution;
    }

    template<typename T>
    void handleExtmethod(T &comp, BhIR *bhir) {
        std::vector<bh_instruction> instr_list;

        for (bh_instruction &instr: bhir->instr_list) {
            auto ext = comp.extmethods.find(instr.opcode);

            if (ext != comp.extmethods.end()) { // Execute the instructions up until now
                BhIR b(std::move(instr_list), bhir->getSyncs());
                comp.execute(&b);
                instr_list.clear(); // Notice, it is legal to clear a moved vector.
                const auto texecution = std::chrono::steady_clock::now();
                ext->second.execute(&instr, nullptr); // Execute the extension method
                stat.time_ext_method += std::chrono::steady_clock::now() - texecution;
            } else {
                instr_list.push_back(instr);
            }
        }

        bhir->instr_list = instr_list;
    }

private:
    void createKernel(std::map<std::string, bool> kernel_config, const std::vector<Block> &block_list) {
        using namespace std;

        // When creating a regular kernels (a block-nest per shared library), we create one kernel at a time
        for (const Block &block: block_list) {
            assert(not block.isInstr());

            // Let's create the symbol table for the kernel
            const SymbolTable symbols(
                    block.getAllInstr(),
                    block.getLoop().getAllNonTemps(),
                    kernel_config["use_volatile"],
                    kernel_config["strides_as_var"],
                    kernel_config["index_as_var"],
                    kernel_config["const_as_var"]
            );

            stat.record(symbols);

            // Let's execute the kernel
            if (not block.isSystemOnly()) { // We can skip this step if the kernel does no computation
                executeKernel({block}, symbols, {});
            }

            // Finally, let's cleanup
            for (bh_base *base: block.getLoop().getAllFrees()) {
                bh_data_free(base);
            }
        }
    }

    void createMonolithicKernel(std::map<std::string, bool> kernel_config, const std::vector<Block> &block_list) {
        using namespace std;

        // When creating a monolithic kernel (all instructions in one shared library), we first combine
        // the instructions and non-temps from each different block and then we create the kernel
        vector<InstrPtr> all_instr;
        set<bh_base *> all_non_temps, all_freed;
        bool kernel_is_computing = false;
        for (const Block &block: block_list) {
            assert(not block.isInstr());
            block.getAllInstr(all_instr);
            block.getLoop().getAllNonTemps(all_non_temps);
            block.getLoop().getAllFrees(all_freed);
            if (not block.isSystemOnly()) {
                kernel_is_computing = true;
            }
        }
        // Let's find the arrays that are allocated and freed between blocks in the kernel
        vector<bh_base *> kernel_temps;
        set<bh_base *> constructors;
        for (const InstrPtr &instr: all_instr) {
            if (instr->constructor) {
                assert(instr->operand[0].base != NULL);
                constructors.insert(instr->operand[0].base);
            } else if (instr->opcode == BH_FREE and util::exist(constructors, instr->operand[0].base)) {
                kernel_temps.push_back(instr->operand[0].base);
                all_non_temps.erase(instr->operand[0].base);
            }
        }

        // Let's create the symbol table for the kernel
        const SymbolTable symbols(
                all_instr,
                all_non_temps,
                kernel_config["use_volatile"],
                kernel_config["strides_as_var"],
                kernel_config["index_as_var"],
                kernel_config["const_as_var"]
        );
        stat.record(symbols);

        // Let's execute the kernel
        if (kernel_is_computing) { // We can skip this step if the kernel does no computation
            executeKernel(block_list, symbols, kernel_temps);
        }

        // Finally, let's cleanup
        for (bh_base *base: all_freed) {
            bh_data_free(base);
        }
    }

private:
    void executeKernel(const std::vector<Block> &block_list,
                       const SymbolTable &symbols,
                       std::vector<bh_base *> kernel_temps) {
        using namespace std;

        // Create the constant vector
        vector<const bh_instruction *> constants;
        constants.reserve(symbols.constIDs().size());
        for (const InstrPtr &instr: symbols.constIDs()) {
            constants.push_back(&(*instr));
        }

        const auto lookup = codegen_cache.get(block_list, symbols);
        if (not lookup.first.empty()) {
            // In debug mode, we check that the cached source code is correct
#ifndef NDEBUG
            stringstream ss;
            writeKernel(block_list, symbols, kernel_temps, lookup.second, ss);
            if (ss.str().compare(lookup.first) != 0) {
                cout << "\nCached source code: \n" << lookup.first;
                cout << "\nReal source code: \n" << ss.str();
                assert(1 == 2);
            }
#endif
            execute(symbols, lookup.first, lookup.second, constants);
        } else {
            const auto tcodegen = chrono::steady_clock::now();
            stringstream ss;
            writeKernel(block_list, symbols, kernel_temps, lookup.second, ss);
            string source = ss.str();
            stat.time_codegen += chrono::steady_clock::now() - tcodegen;

            execute(symbols, source, lookup.second, constants);
            codegen_cache.insert(std::move(source), block_list, symbols);
        }
    }
};

}
} // namespace
