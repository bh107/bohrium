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
#include <vector>
#include <set>

#include <jitk/engines/engine_cpu.hpp>

#include <bh_config_parser.hpp>
#include <jitk/statistics.hpp>
#include <jitk/apply_fusion.hpp>

#include <bh_view.hpp>
#include <bh_component.hpp>
#include <bh_instruction.hpp>
#include <bh_main_memory.hpp>

using namespace std;

namespace bohrium {
namespace jitk {

namespace {
std::vector<InstrPtr> order_sweep_by_origin_id(const std::set<InstrPtr> &sweep_set) {
    vector<InstrPtr> ret;
    ret.reserve(sweep_set.size());
    std::copy(sweep_set.begin(),  sweep_set.end(), std::back_inserter(ret));
    std::sort(ret.begin(), ret.end(),
              [](const InstrPtr & a, const InstrPtr & b) -> bool
              {
                  return a->origin_id > b->origin_id;
              });
    return ret;
}


// Adds identity blocks before sweeping blocks
void add_identity_block(LoopB &loop, int64_t &origin_count) {
    vector<Block> ret;
    for (Block &block: loop._block_list) {
        if (block.isInstr()) {
            ret.push_back(block);
            continue;
        }
        add_identity_block(block.getLoop(), origin_count);

        const auto ordered_sweeps = order_sweep_by_origin_id(block.getLoop().getSweeps());
        for (const InstrPtr &sweep_instr: ordered_sweeps) {
            bh_instruction identity_instr(BH_IDENTITY, {sweep_instr->operand[0]});
            identity_instr.operand.resize(2);
            identity_instr.operand[1].base = nullptr;
            identity_instr.constant = sweep_identity(sweep_instr->opcode, sweep_instr->operand[0].base->type);
            identity_instr.origin_id = origin_count++;
            identity_instr.constructor = sweep_instr->constructor;
            // We have to manually set the sweep axis of an accumulate output to 1. The backend will execute
            // the for-loop in serial thus only the first element should be the identity.
            if (bh_opcode_is_accumulate(sweep_instr->opcode)) {
                identity_instr.operand[0].shape[sweep_instr->sweep_axis()] = 1;
            }

            if (loop.rank == -1 and bh_is_scalar(&sweep_instr->operand[0])) {
                ret.emplace_back(identity_instr, 0);
            } else if (loop.rank == sweep_instr->operand[0].ndim - 1) {
                ret.emplace_back(identity_instr, sweep_instr->operand[0].ndim);
            } else {
                // Let's create and add the identity loop to `ret`
                vector<InstrPtr> single_instr = {std::make_shared<const bh_instruction>(identity_instr)};
                ret.push_back(create_nested_block(single_instr, loop.rank+1));
            }

            bh_instruction sweep_instr_updated{*sweep_instr};
            sweep_instr_updated.constructor = false;
            block.getLoop().replaceInstr(sweep_instr, sweep_instr_updated);
            block.getLoop().metadataUpdate();
        }
        ret.push_back(block);
    }
    loop._block_list = ret;
    loop.metadataUpdate();
}
}

void EngineCPU::handleExecution(BhIR *bhir) {

    const auto texecution = chrono::steady_clock::now();

    map<string, bool> kernel_config = {
            {"strides_as_var", comp.config.defaultGet<bool>("strides_as_var", true)},
            {"index_as_var",   comp.config.defaultGet<bool>("index_as_var", true)},
            {"const_as_var",   comp.config.defaultGet<bool>("const_as_var", true)},
            {"use_volatile",   comp.config.defaultGet<bool>("use_volatile", false)}
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
    if (comp.config.defaultGet<bool>("array_contraction", true)) {
        setConstructorFlag(instr_list);
    } else {
        for (bh_instruction *instr: instr_list) {
            instr->constructor = false;
        }
    }

    // Let's get the block list
    vector<Block> block_list = get_block_list(instr_list, comp.config, fcache, stat, false);

    LoopB kernel{-1, 1, {block_list}};

//    cout << "before add_identity_block " << endl;
//    cout << kernel._block_list;
    int64_t origin_count = 100000;
    add_identity_block(kernel, origin_count);
//    cout << kernel._block_list;

    createKernel(kernel_config, Block{kernel});

    stat.time_total_execution += chrono::steady_clock::now() - texecution;
}

void EngineCPU::handleExtmethod(BhIR *bhir){
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

void EngineCPU::createKernel(std::map<std::string, bool> &kernel_config, const Block &block) {
    assert(block.rank() == -1);
    assert(not block.isInstr());

    // Let's create the symbol table for the kernel
    const SymbolTable symbols(block.getLoop(),
            kernel_config["use_volatile"],
            kernel_config["strides_as_var"],
            kernel_config["index_as_var"],
            kernel_config["const_as_var"]
    );

    stat.record(symbols);

    if (not block.isSystemOnly()) { // We can skip this step if the kernel does no computation
        // Create the constant vector
        vector<const bh_instruction *> constants;
        constants.reserve(symbols.constIDs().size());
        for (const InstrPtr &instr: symbols.constIDs()) {
            constants.push_back(&(*instr));
        }

        const auto lookup = codegen_cache.lookup(block.getLoop(), symbols);
        if (not lookup.first.empty()) {
            // In debug mode, we check that the cached source code is correct
            #ifndef NDEBUG
                stringstream ss;
                writeKernel(block.getLoop(), symbols, {}, lookup.second, ss);
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
            writeKernel(block.getLoop(), symbols, {}, lookup.second, ss);
            string source = ss.str();
            stat.time_codegen += chrono::steady_clock::now() - tcodegen;

            execute(symbols, source, lookup.second, constants);
            codegen_cache.insert(std::move(source), block.getLoop(), symbols);
        }
    }

    // Finally, let's cleanup
    for (bh_base *base: block.getLoop().getAllFrees()) {
        bh_data_free(base);
    }
}

}
}
