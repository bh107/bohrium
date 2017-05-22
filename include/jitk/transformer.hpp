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

/* A collection of functions that transforms Blocks of Instructions */

#include <iostream>

#include <bh_instruction.hpp>
#include <jitk/block.hpp>

namespace bohrium {
namespace jitk {

// Transpose blocks such that reductions gets as innermost as possible
void push_reductions_inwards(std::vector<Block> &block_list);

// Splits the 'block_list' in order to achieve a minimum amount of threading (if possible)
void split_for_threading(std::vector<Block> &block_list, uint64_t min_threading=1000, uint64_t cur_threading=0);

// Collapses redundant axes within the 'block_list'
void collapse_redundant_axes(std::vector<Block> &block_list);

} // jitk
} // bohrium

#endif
