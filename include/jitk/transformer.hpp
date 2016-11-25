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

#ifndef __BH_JITK_TRANSFORMER_HPP
#define __BH_JITK_TRANSFORMER_HPP

/* A collection of functions that transforms Blocks that Instructions */

#include <iostream>

#include <bh_instruction.hpp>
#include <jitk/block.hpp>

namespace bohrium {
namespace jitk {

// Swap the two axises, 'axis1' and 'axis2', in all instructions in 'instr_list'
std::vector<InstrPtr> swap_axis(const std::vector<InstrPtr> &instr_list, int64_t axis1, int64_t axis2);

// Swap the 'parent' block with its 'child' block.
// NB: 'child' must point to a block in 'parent._block_list'
std::vector<Block> swap_blocks(const LoopB &parent, const LoopB *child);

// Find a loop block within 'parent' that it make sense to swappable
const LoopB *find_swappable_sub_block(const LoopB &parent);

// Transpose blocks such that reductions gets as innermost as possible
std::vector<Block> push_reductions_inwards(const std::vector<Block> &block_list);

// Splits the 'block_list' in order to achieve a minimum amount of threading (if possible)
std::vector<Block> split_for_threading(const std::vector<Block> &block_list, uint64_t min_threading,
                                       uint64_t cur_threading=0);

} // jitk
} // bohrium

#endif
