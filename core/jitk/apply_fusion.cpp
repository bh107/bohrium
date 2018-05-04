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

namespace {
// Apply the pre-fuser (i.e. fuse an instruction list to a block list specified by the name 'transformer_name'
vector<Block> apply_pre_fusion(const ConfigParser &config, const vector<bh_instruction*> &instr_list,
                               const string &transformer_name) {
    // The include the 'singleton' and 'pre_fuser_lossy' names for legacy support
    if (transformer_name == "none" or transformer_name == "singleton") {
        return fuser_singleton(instr_list);
    } else if (transformer_name == "lossy" or transformer_name == "pre_fuser_lossy"){
        return pre_fuser_lossy(instr_list);
    } else {
        cout << "Unknown pre-fuser: \"" <<  transformer_name << "\"" << endl;
        throw runtime_error("Unknown pre-fuser!");
    }
}

// Apply the list of transformer specified by the names in 'transformer_names'
// 'avoid_rank0_sweep' will avoid fusion of sweeped and non-sweeped blocks at the root level
void apply_transformers(const ConfigParser &config, vector<Block> &block_list, const vector<string> &transformer_names,
                        bool avoid_rank0_sweep) {

    for(auto it = transformer_names.begin(); it != transformer_names.end(); ++it) {
        if (*it == "push_reductions_inwards") {
            push_reductions_inwards(block_list);
        } else if (*it == "split_for_threading") {
            split_for_threading(block_list);
        } else if (*it == "collapse_redundant_axes") {
            collapse_redundant_axes(block_list);
        } else if (*it == "serial") {
            fuser_serial(block_list, avoid_rank0_sweep);
        } else if (*it == "breadth_first") {
            fuser_breadth_first(block_list, avoid_rank0_sweep);
        } else if (*it == "reshapable_first") {
            fuser_reshapable_first(block_list, avoid_rank0_sweep);
        } else if (*it == "greedy") {
            fuser_greedy(config, block_list, avoid_rank0_sweep);
        } else {
            cout << "Unknown transformer: \"" << *it << "\"" << endl;
            throw runtime_error("Unknown transformer!");
        }
    }
}

// For better codegen cache utilization, we make sure that sweep instructions comes in a consisten order
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


// Help function that adds identity blocks before sweeping blocks.
// This version update a block rather then return a kernel list (see the function belove)
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

// Help function that adds identity blocks before sweeping blocks.
// This version creates kernels for each identity+sweep block set
vector<LoopB> add_identity_block(vector<Block> &block_list, int64_t &origin_count) {
    vector<LoopB> ret;
    for (Block &block: block_list) {
        assert(not block.isInstr());
        add_identity_block(block.getLoop(), origin_count);

        LoopB kernel{-1, 1};

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

            if (bh_is_scalar(&sweep_instr->operand[0])) {
                kernel._block_list.emplace_back(identity_instr, 0);
            } else {
                // Let's create and add the identity loop to `ret`
                vector<InstrPtr> single_instr = {std::make_shared<const bh_instruction>(identity_instr)};
                kernel._block_list.emplace_back(create_nested_block(single_instr, 0));
            }

            bh_instruction sweep_instr_updated{*sweep_instr};
            sweep_instr_updated.constructor = false;
            block.getLoop().replaceInstr(sweep_instr, sweep_instr_updated);
            block.getLoop().metadataUpdate();
        }
        kernel._block_list.push_back(block);
        kernel.metadataUpdate();
        ret.emplace_back(std::move(kernel));
    }
    return ret;
}

// Help functions that create a list of block nest (each nest starting a rank 0) based on `instr_list`
// 'avoid_rank0_sweep' will avoid fusion of sweeped and non-sweeped blocks at the root level
vector<Block> get_block_list(const vector<bh_instruction*> &instr_list, const ConfigParser &config,
                             FuseCache &fcache, Statistics &stat, bool avoid_rank0_sweep) {
    vector<Block> block_list;
    bool hit;
    tie(block_list, hit) = fcache.get(instr_list);
    if (not hit) {
        const auto tpre_fusion = chrono::steady_clock::now();
        stat.num_instrs_into_fuser += instr_list.size();
        // Let's fuse the 'instr_list' into blocks
        // We start with the pre_fuser
        block_list = apply_pre_fusion(config, instr_list, config.defaultGet("pre_fuser", string("pre_fuser_lossy")));
        stat.num_blocks_out_of_fuser += block_list.size();
        const auto tfusion = chrono::steady_clock::now();
        stat.time_pre_fusion += tfusion - tpre_fusion;
        // Then we fuse fully
        apply_transformers(config, block_list, config.defaultGetList("fuser_list", {"greedy"}), avoid_rank0_sweep);
        stat.time_fusion += chrono::steady_clock::now() - tfusion;
        fcache.insert(instr_list, block_list);
    }

    // Pretty printing the block
    if (config.defaultGet<bool>("graph", false)) {
        static int dcount = 0;
        {
            graph::DAG dag = graph::from_block_list(block_list);
            graph::pprint(dag, "dag", avoid_rank0_sweep, dcount);
        }
        {
            graph::DAG dag = graph::from_block_list(apply_pre_fusion(config, instr_list, "singleton"));
            graph::pprint(dag, "dag_singleton", avoid_rank0_sweep, dcount);
        }
        {
            graph::DAG dag = graph::from_block_list(apply_pre_fusion(config, instr_list, "lossy"));
            graph::pprint(dag, "dag_lossy", avoid_rank0_sweep, dcount);
        }
        ++dcount;
    }

    // Full block validation
    #ifndef NDEBUG
        for (const Block &b: block_list) {
            if (b.isInstr()) {
                assert(b.rank() == 1);
            }
            assert(b.validation());
        }
    #endif
    return block_list;
}
}


vector<LoopB> get_kernel_list(const vector<bh_instruction*> &instr_list, const ConfigParser &config,
                              FuseCache &fcache, Statistics &stat, bool avoid_rank0_sweep, bool monolithic) {
    // Assign origin ids to all instructions starting at zero.
    int64_t origin_count = 0;
    for (bh_instruction *instr: instr_list) {
        instr->origin_id = origin_count++;
    }

    vector<Block> block_list = get_block_list(instr_list, config, fcache, stat, avoid_rank0_sweep);

    vector<LoopB> ret;
    if (avoid_rank0_sweep) {
        for (Block &b: block_list) {
            if (b.isInstr() or not b.getLoop()._sweeps.empty()) {
                ret.emplace_back(LoopB{-1, 1, {std::move(b)}});
            } else {
                LoopB kernel{-1, 1, {std::move(b)}};
                add_identity_block(kernel, origin_count);
                ret.emplace_back(std::move(kernel));
            }
        }
    } else if (monolithic) {
        LoopB kernel{-1, 1, {std::move(block_list)}};
        add_identity_block(kernel, origin_count);
        ret = {std::move(kernel)};
    } else {
        ret = add_identity_block(block_list, origin_count);
    }
    return ret;
}

} // jitk
} // bohrium
