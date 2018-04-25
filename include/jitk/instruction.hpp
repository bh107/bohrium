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

#include <iostream>
#include <bh_instruction.hpp>
#include <jitk/block.hpp>
#include <jitk/scope.hpp>

namespace bohrium {
namespace jitk {

// Write the source code of an instruction (set 'opencl' for OpenCL specific output)
void write_instr(const Scope &scope, const bh_instruction &instr, std::stringstream &out, bool opencl = false);

// Write the neutral value of a reduction
void write_reduce_identity(bh_opcode opcode, bh_type dtype, std::stringstream &out);

// Return the identity value of an sweep operation
bh_constant sweep_identity(bh_opcode opcode, bh_type dtype);

// Removes syncs and frees from 'instr_list' that are never used in a computation.
// 'syncs' and 'frees' are the sets of arrays that were removed.
std::vector<bh_instruction*> remove_non_computed_system_instr(std::vector<bh_instruction> &instr_list,
                                                              std::set<bh_base *> &frees);

// Reshape 'instr' to match 'size_of_rank_dim' at the 'rank' dimension.
// The dimensions from zero to 'rank-1' are untouched.
InstrPtr reshape_rank(const InstrPtr &instr, int rank, int64_t size_of_rank_dim);

} // jitk
} // bohrium
