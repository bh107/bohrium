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
    const vector <jitk::Block> block_list = get_block_list(instr_list, comp.config, fcache, stat, false);

    if (comp.config.defaultGet<bool>("monolithic", false)) {
        createMonolithicKernel(kernel_config, block_list);
    } else {
        createKernel(kernel_config, block_list);
    }
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

}
}
