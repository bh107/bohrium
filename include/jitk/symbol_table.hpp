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

#include <map>
#include <vector>
#include <string>
#include <sstream>

#include <bh_view.hpp>
#include <bh_util.hpp>

#include <jitk/block.hpp>

namespace bohrium {
namespace jitk {


// Compare class for the OffsetAndStrides sets and maps
struct OffsetAndStrides_less {
    // This compare is the same as view compare ('v1 < v2') but ignoring their bases
    bool operator() (const bh_view& v1, const bh_view& v2) const {
        if (v1.ndim < v2.ndim) return true;
        if (v2.ndim < v1.ndim) return false;
        if (v1.start < v2.start) return true;
        if (v2.start < v1.start) return false;
        for (int64_t i = 0; i < v1.ndim; ++i) {
            if (v1.stride[i] < v2.stride[i]) return true;
            if (v2.stride[i] < v1.stride[i]) return false;
            if (v1.shape[i] < v2.shape[i]) return true;
            if (v2.shape[i] < v1.shape[i]) return false;
        }
        return false;
    }
    bool operator() (const bh_view* v1, const bh_view* v2) const {
        return (*this)(*v1, *v2);
    }
};

// Compare class for the constant_map
struct Constant_less {
    // This compare tje 'origin_id' member of the instructions
    bool operator() (const InstrPtr &i1, const InstrPtr& i2) const {
        return i1->origin_id < i2->origin_id;
    }
};

// The SymbolTable class contains all array meta date needed for a JIT kernel.
class SymbolTable {
private:
    std::map<const bh_base*, size_t> _base_map; // Mapping a base to its ID
    std::map<bh_view, size_t> _view_map; // Mapping a view to its ID
    std::map<bh_view, size_t, OffsetAndStrides_less> _idx_map; // Mapping a index (of an array) to its ID
    std::map<bh_view, size_t, OffsetAndStrides_less> _offset_strides_map; // Mapping a offset-and-strides to its ID
    std::vector<const bh_view*> _offset_stride_views; // Vector of all offset-and-stride views
    std::set<InstrPtr, Constant_less> _constant_set; // Set of instructions to a constant ID (Order by `origin_id`)
    std::set<bh_base*> _array_always; // Set of base arrays that should always be arrays
    std::vector<bh_base*> _params; // Vector of non-temporary arrays, which are the in-/out-puts of the JIT kernel
    bool _useRandom; // Flag: is any instructions using random?

public:
    // Should we declare scalar variables using the volatile keyword?
    const bool use_volatile;
    // Should we use start and strides as variables?
    const bool strides_as_var;
    // Should we save index calculations in variables?
    const bool index_as_var;
    // Should we use constants as variables?
    const bool const_as_var;

    SymbolTable(const LoopB &kernel, bool use_volatile, bool strides_as_var, bool index_as_var, bool const_as_var);

    // Get the ID of 'base', throws exception if 'base' doesn't exist
    size_t baseID(const bh_base *base) const {
        return _base_map.at(base);
    }
    // Get total number of base arrays
    size_t getNumBaseArrays() const {
        return _base_map.size();
    }
    // Get the ID of 'view', throws exception if 'view' doesn't exist
    size_t viewID(const bh_view &view) const {
        return _view_map.at(view);
    }
    // Get the ID of 'index', throws exception if 'index' doesn't exist
    size_t idxID(const bh_view &index) const {
        return _idx_map.at(index);
    }
    // Check if 'index' exist
    bool existIdxID(const bh_view &index) const {
        return util::exist(_idx_map, index);
    }
    // Get the offset-and-strides ID of 'view', throws exception if 'view' doesn't exist
    size_t offsetStridesID(const bh_view &view) const {
        return _offset_strides_map.at(view);
    }
    bool existOffsetStridesID(const bh_view &view) const {
        return util::exist(_offset_strides_map,view);
    }
    const std::vector<const bh_view*> &offsetStrideViews() const {
        return _offset_stride_views;
    }
    // Get the set of constants
    const std::set<InstrPtr, Constant_less> &constIDs() const {
        return _constant_set;
    };
    // Get the ID of the constant within 'instr', which is the number it appear in the set of constants.
    // Or returns -1 when 'instr' has no ID
    int64_t constID(const bh_instruction &instr) const {
        assert(instr.origin_id >= 0);
        size_t count=0;
        // Since the size of '_constant_set' is small (way below a 1000), we simply do a linear search
        for (const InstrPtr &i: _constant_set) {
            count++;
            if (i->origin_id == instr.origin_id)
                return count;
        }
        return -1;
    }
    // Return true when 'base' should always be an array
    bool isAlwaysArray(const bh_base *base) const {
        return util::exist_nconst(_array_always, base);
    }
    // Return non-temporary arrays, which are the in-/out-puts of the JIT kernel, in the order of their IDs
    const std::vector<bh_base*> &getParams() const {
        return _params;
    }
    // Is any instructions use the random library?
    bool useRandom() const {
        return _useRandom;
    }
};

} // jitk
} // bohrium
