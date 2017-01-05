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

#ifndef __BH_JITK_INSTRUCTION_HPP
#define __BH_JITK_INSTRUCTION_HPP

#include <iostream>

#include <bh_instruction.hpp>

#include "base_db.hpp"

namespace bohrium {
namespace jitk {


// Write the source code of an instruction (set 'opencl' for OpenCL specific output)
void write_instr(const BaseDB &base_ids, const bh_instruction &instr, std::stringstream &out, bool opencl=false);

// Write the array subscription, e.g. A[2+i0*1+i1*10], but ignore the loop-variant of 'hidden_axis' if it isn't 'BH_MAXDIM'
void write_array_subscription(const bh_view &view, std::stringstream &out, int hidden_axis = BH_MAXDIM,
                              const std::pair<int, int> axis_offset = std::make_pair(BH_MAXDIM, 0));

// Return true when 'opcode' has a neutral initial reduction value
bool has_reduce_identity(bh_opcode opcode);

// Write the neutral value of a reduction
void write_reduce_identity(bh_opcode opcode, bh_type dtype, std::stringstream &out);

// Removes syncs and frees from 'instr_list' that are never used in a computation.
// 'syncs' and 'frees' are the sets of arrays that were removed.
std::vector<bh_instruction*> remove_non_computed_system_instr(std::vector<bh_instruction> &instr_list,
                                                              std::set<bh_base *> &syncs, std::set<bh_base *> &frees);

} // jitk
} // bohrium

#endif
