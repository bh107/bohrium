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

#ifndef __BH_JITK_FUSER_HPP
#define __BH_JITK_FUSER_HPP

#include <set>
#include <vector>

#include <jitk/block.hpp>
#include <bh_instruction.hpp>


namespace bohrium {
namespace jitk {

// Creates a block list based on the 'instr_list' where each instruction gets its own nested block
// 'news' is the set of instructions in 'instr_list' that initiates new base arrays
// NB: this function might reshape the instructions in 'instr_list'
std::vector<Block> fuser_singleton(std::vector<bh_instruction*> &instr_list, const std::set<bh_instruction*> &news);

// Fuses 'block_list' in a serial naive manner
// 'news' is the set of instructions in 'instr_list' that initiates new base arrays
std::vector<Block> fuser_serial(const std::vector<Block> &block_list, const std::set<bh_instruction*> &news);

// Fuses 'block_list' in a topological breadth first manner
// 'news' is the set of instructions in 'instr_list' that initiates new base arrays
std::vector<Block> fuser_breadth_first(const std::vector<Block> &block_list, const std::set<bh_instruction *> &news);

// Fuses 'block_list' in a topological manner prioritizing fusion of reshapable blocks
// 'news' is the set of instructions in 'instr_list' that initiates new base arrays
std::vector<Block> fuser_reshapable_first(const std::vector<Block> &block_list, const std::set<bh_instruction *> &news);

} // jit
} // bohrium

#endif
