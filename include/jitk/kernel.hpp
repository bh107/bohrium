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

#ifndef __BH_JITK_KERNEL_HPP
#define __BH_JITK_KERNEL_HPP

#include <set>
#include <vector>

#include <jitk/block.hpp>
#include <jitk/base_db.hpp>
#include <jitk/statistics.hpp>

namespace bohrium {
namespace jitk {

class Kernel {
private:
    // Do the kernel use random?
    bool _useRandom;
    // Arrays freed
    std::set<bh_base*> _frees;
    // Arrays sync'ed
    std::set<bh_base*> _syncs;
    // Non-temporary arrays, which makes up the parameters to a kernel
    std::vector<bh_base*> _non_temps;

public:

    // The loop block that makes up this kernel
    const LoopB &block;

    // Constructor
    Kernel(const LoopB &block);

    // Do the kernel use random?
    bool useRandom() const {
        return _useRandom;
    }

    // Return the freed arrays
    const std::set<bh_base*> &getFrees() const {
        return _frees;
    }

    // Return the sync'ed arrays
    const std::set<bh_base*> &getSyncs() const {
        return _syncs;
    }

    // Return the non-temporary arrays
    const std::vector<bh_base*> &getNonTemps() const {
        return _non_temps;
    }

    // Return all instructions in the kernel
    void getAllInstr(std::vector<InstrPtr> &out) const {
        block.getAllInstr(out);
    }
    std::vector<InstrPtr> getAllInstr() const {
        std::vector<InstrPtr> ret;
        getAllInstr(ret);
        return ret;
    }

    // Return all temporary arrays in the kernel
    void getAllTemps(std::set<bh_base*> &out) const {
        block.getAllTemps(out);
    }
    std::set<bh_base*> getAllTemps() const {
        std::set<bh_base*> ret;
        getAllTemps(ret);
        return ret;
    }

    // Returns all offset-and-strides of all non-temporary arrays in the order the appear in the instruction list
    std::vector<const bh_view*> getOffsetAndStrides() const {
        std::vector<const bh_view*> ret;
        {
            std::set<bh_view, OffsetAndStrides_less> offset_strides_set;
            for (const InstrPtr instr: getAllInstr()) {
                for (const bh_view *view: instr->get_views()) {
                    if (util::exist_linearly(getNonTemps(), view->base) and
                        not util::exist(offset_strides_set, *view)) {
                        ret.push_back(view);
                        offset_strides_set.insert(*view);
                    }
                }
            }
        }
        return ret;
    }
};

// Create a new Kernel object including statistics and verbosity
Kernel create_kernel_object(const Block &block, const bool verbose, Statistics &stat);

} // jit
} // bohrium

#endif
