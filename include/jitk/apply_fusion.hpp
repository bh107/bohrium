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

#ifndef __BH_JITK_APPLY_FUSION_HPP
#define __BH_JITK_APPLY_FUSION_HPP

#include <iostream>

#include <bh_instruction.hpp>
#include <jitk/block.hpp>
#include <jitk/fuser.hpp>
#include <jitk/transformer.hpp>

namespace bohrium {
namespace jitk {


// Apply the pre-fuser (i.e. fuse an instruction list to a block list specified by the name 'transformer_name'
std::vector<Block> apply_pre_fusion(const std::vector<bh_instruction*> &instr_list,
                                    const std::string &transformer_name) {

    if (transformer_name == "singleton") {
        return fuser_singleton(instr_list);
    } else if (transformer_name == "pre_fuser_lossy"){
        return pre_fuser_lossy(instr_list);
    } else {
        std::cout << "Unknown pre-fuser: \"" <<  transformer_name << "\"" << std::endl;
        throw std::runtime_error("Unknown pre-fuser!");
    }
}

// Apply the list of tranformers specified by the names in 'transformer_names'
void apply_transformers(std::vector<Block> &block_list, const std::vector<std::string> &transformer_names) {

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
            std::cout << "Unknown transformer: \"" << *it << "\"" << std::endl;
            throw std::runtime_error("Unknown transformer!");
        }
    }
}

} // jitk
} // bohrium

#endif
