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

#ifndef __BH_VE_UNI_KERNEL_HPP
#define __BH_VE_UNI_KERNEL_HPP

#include <set>
#include <vector>

#include "block.hpp"

namespace bohrium {

class Kernel {
public:
    Kernel() {}

    std::vector <Block> block_list;
    bool useRandom = false;

    std::vector<bh_base*> non_temps;
    std::set<bh_base*> frees;
    std::set<bh_base*> sync;

    // Return all instructions in the kernel
    void getAllInstr(std::vector<bh_instruction*> &out) const {
        for(const Block &b: block_list) {
            b.getAllInstr(out);
        }
    }
    std::vector<bh_instruction*> getAllInstr() const {
        std::vector<bh_instruction *> ret;
        getAllInstr(ret);
        return ret;
    }

    // Return all temporary arrays in the kernel
    void getAllTemps(std::set<bh_base*> &out) const {
        for(const Block &b: block_list) {
            b.getAllTemps(out);
        }
    }
    std::set<bh_base*> getAllTemps() const {
        std::set<bh_base*> ret;
        getAllTemps(ret);
        return ret;
    }

    // Insert a non-temporary array
    void insertNonTemp(bh_base *base) {
        if (std::find(non_temps.begin(), non_temps.end(), base) == non_temps.end())
            non_temps.push_back(base);
    }
};

} // bohrium

#endif
