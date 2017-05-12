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

#include <cassert>

#include <jitk/apply_fusion.hpp>
#include <jitk/graph.hpp>

using namespace std;

namespace bohrium {
namespace jitk {

vector<Block> apply_pre_fusion(const vector<bh_instruction*> &instr_list,
                                    const string &transformer_name) {

    if (transformer_name == "singleton") {
        return fuser_singleton(instr_list);
    } else if (transformer_name == "pre_fuser_lossy"){
        return pre_fuser_lossy(instr_list);
    } else {
        cout << "Unknown pre-fuser: \"" <<  transformer_name << "\"" << endl;
        throw runtime_error("Unknown pre-fuser!");
    }
}

void apply_transformers(vector<Block> &block_list, const vector<string> &transformer_names) {

    for(auto it = transformer_names.begin(); it != transformer_names.end(); ++it) {
        if (*it == "push_reductions_inwards") {
            push_reductions_inwards(block_list);
        } else if (*it == "split_for_threading") {
            split_for_threading(block_list);
        } else if (*it == "collapse_redundant_axes") {
            collapse_redundant_axes(block_list);
        } else if (*it == "serial") {
            fuser_serial(block_list);
        } else if (*it == "breadth_first") {
            fuser_breadth_first(block_list);
        } else if (*it == "reshapable_first") {
            fuser_reshapable_first(block_list);
        } else if (*it == "greedy") {
            fuser_greedy(block_list);
        } else {
            cout << "Unknown transformer: \"" << *it << "\"" << endl;
            throw runtime_error("Unknown transformer!");
        }
    }
}

vector<Block> get_block_list(const vector<bh_instruction*> &instr_list, const ConfigParser &config,
                                  FuseCache &fcache, Statistics &stat) {

    vector<Block> block_list;

    // Assign origin ids to all instructions starting at zero.
    int64_t count = 0;
    for (bh_instruction *instr: instr_list) {
        instr->origin_id = count++;
    }

    bool hit;
    tie(block_list, hit) = fcache.get(instr_list);
    if (not hit) {
        const auto tpre_fusion = chrono::steady_clock::now();
        stat.num_instrs_into_fuser += instr_list.size();
        // Let's fuse the 'instr_list' into blocks
        // We start with the pre_fuser
        block_list = apply_pre_fusion(instr_list, config.defaultGet("pre_fuser", string("pre_fuser_lossy")));
        stat.num_blocks_out_of_fuser += block_list.size();
        const auto tfusion = chrono::steady_clock::now();
        stat.time_pre_fusion += tfusion - tpre_fusion;
        // Then we fuse fully
        apply_transformers(block_list, config.defaultGetList("fuser_list", {"greedy"}));
        stat.time_fusion += chrono::steady_clock::now() - tfusion;
        fcache.insert(instr_list, block_list);
    }

    // Pretty printing the block
    if (config.defaultGet<bool>("graph", false)) {
        graph::DAG dag = graph::from_block_list(block_list);
        graph::pprint(dag, "dag");
    }

    return block_list;
}

} // jitk
} // bohrium
