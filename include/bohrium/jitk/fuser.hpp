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

#include <set>
#include <vector>

#include <bohrium/jitk/block.hpp>
#include <bohrium/bh_config_parser.hpp>
#include <bohrium/bh_instruction.hpp>

namespace bohrium {
namespace jitk {

/// In order to avoid duplicate calls to `ConfigParser`, we this struct
struct FusionConfig {
    /// Will avoid fusion of sweeped and non-sweeped blocks at the root level
    bool avoid_rank0_sweep;
    // Flag to place all kernels into one
    bool monolithic;
    /// The pre-fuser to use
    std::string pre_fuser;
    /// List of fusers to use
    std::vector<std::string> fuser_list;
    /// When using the greedy fuser, when exceeding `greedy_threshold` nodes fuser_reshapable_first() is used instead.
    uint64_t greedy_threshold;
    /// Dump fusion graph
    bool graph;

    FusionConfig(const ConfigParser &config, bool avoid_rank0_sweep) :
            avoid_rank0_sweep(avoid_rank0_sweep),
            monolithic(config.defaultGet<bool>("monolithic", false)),
            pre_fuser(config.defaultGet("pre_fuser", std::string("lossy"))),
            fuser_list(config.defaultGetList("fuser_list", {"greedy"})),
            greedy_threshold(config.defaultGet<uint64_t>("greedy_threshold", 10000)),
            graph(config.defaultGet<bool>("graph", false)) {}
};

// Creates an instruction of 'InstrPtr' from an instruction list with all noop operations removed
std::vector<InstrPtr> simplify_instr_list(const std::vector<bh_instruction *> &instr_list);

// Creates a block list based on the 'instr_list' where only fully fusible instructions are fused
std::vector<Block> pre_fuser_lossy(const std::vector<bh_instruction *> &instr_list);

// Creates a block list based on the 'instr_list' where each instruction gets its own nested block
// NB: this function might reshape the instructions in 'instr_list'
std::vector<Block> fuser_singleton(const std::vector<bh_instruction *> &instr_list);

// Fuses 'block_list' in a serial naive manner
// 'avoid_rank0_sweep' will avoid fusion of sweeped and non-sweeped blocks at the root level
void fuser_serial(std::vector<Block> &block_list, bool avoid_rank0_sweep);

// Fuses 'block_list' in a topological breadth first manner
// 'avoid_rank0_sweep' will avoid fusion of sweeped and non-sweeped blocks at the root level
void fuser_breadth_first(std::vector<Block> &block_list, bool avoid_rank0_sweep);

// Fuses 'block_list' in a topological manner prioritizing fusion of reshapable blocks
// 'avoid_rank0_sweep' will avoid fusion of sweeped and non-sweeped blocks at the root level
void fuser_reshapable_first(std::vector<Block> &block_list, bool avoid_rank0_sweep);

// Fuses 'block_list' greedily
void fuser_greedy(const FusionConfig &config, std::vector<Block> &block_list);

} // jit
} // bohrium
