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
#include <chrono>

#include <bohrium/bh_instruction.hpp>
#include <bohrium/bh_config_parser.hpp>
#include <bohrium/jitk/block.hpp>
#include <bohrium/jitk/fuser.hpp>
#include <bohrium/jitk/transformer.hpp>
#include <bohrium/jitk/fuser_cache.hpp>
#include <bohrium/jitk/statistics.hpp>

namespace bohrium {
namespace jitk {

/** Create a kernel list based on 'instr_list' and what is in the 'config' and 'fcache'
 * Notice, this function inject identity instructions before sweep instructions
 *
 * @param instr_list The instruction list to base the kernel list on
 * @param config     The fusion config
 * @param fcache     The fuse cache
 * @param stat       Statistics
 * @return Return a list of kernels
 */
std::vector<LoopB>
get_kernel_list(const std::vector<bh_instruction *> &instr_list, const FusionConfig &config, FuseCache &fcache,
                Statistics &stat);

} // jitk
} // bohrium
