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

#include <bh_util.hpp>
#include <jitk/symbol_table.hpp>
#include <jitk/view.hpp>

using namespace std;

namespace bohrium {
namespace jitk {

SymbolTable::SymbolTable(const LoopB &kernel,
                         bool use_volatile,
                         bool strides_as_var,
                         bool index_as_var,
                         bool const_as_var) : _useRandom(false),
                                              use_volatile(use_volatile),
                                              strides_as_var(strides_as_var),
                                              index_as_var(index_as_var),
                                              const_as_var(const_as_var) {

    // NB: by assigning the IDs in the order they appear in the 'instr_list',
    //     the kernels can better be reused
    const std::vector <InstrPtr> instr_list = kernel.getAllInstr();
    for (const InstrPtr &instr: instr_list) {
        for (const bh_view *view: instr->get_views()) {
            _base_map.insert(std::make_pair(view->base, _base_map.size()));
            _view_map.insert(std::make_pair(*view, _view_map.size()));
            if (index_as_var) {
                _idx_map.insert(std::make_pair(*view, _idx_map.size()));
            }
            _offset_strides_map.insert(std::make_pair(*view, _offset_strides_map.size()));
        }
        if (const_as_var) {
            assert(instr->origin_id >= 0);
            if (instr->has_constant()) {
                _constant_set.insert(instr);
            }
        }
        // Since accumulate accesses the previous index, it should always be an array
        if (bh_opcode_is_accumulate(instr->opcode)) {
            _array_always.insert(instr->operand[0].base);
        } else if (instr->opcode == BH_GATHER) { // Gather accesses the input arbitrarily
            if (not bh_is_constant(&instr->operand[1])) {
                _array_always.insert(instr->operand[1].base);
            }
          // Scatter accesses the output arbitrarily
        } else if (instr->opcode == BH_SCATTER or instr->opcode == BH_COND_SCATTER) {
            _array_always.insert(instr->operand[0].base);
        } else if (instr->opcode == BH_RANDOM) {
            _useRandom = true;
        }
    }
    
    // Add frees to the base map since the are not in `kernel.getAllInstr()`
    for (const bh_base *base: kernel.getAllFrees()) {
        _base_map.insert(std::make_pair(base, _base_map.size()));
    }

    // Find bases that are the parameters to the JIT kernel, which are non-temporary arrays not
    // already in `_params`. NB: the order of `_params` matches the order of the array IDs
    {
        auto non_temp_arrays = kernel.getAllNonTemps();
        non_temp_arrays.insert(_array_always.begin(), _array_always.end());
        for (const InstrPtr &instr: instr_list) {
            for (const bh_view &v: instr->operand) {
                if (not bh_is_constant(&v) and util::exist(non_temp_arrays, v.base)) {
                    if (not util::exist_linearly(_params, v.base)) {
                        _params.push_back(v.base);
                    }
                }
            }
        }
    }
    if (strides_as_var) {
        _offset_stride_views.resize(_offset_strides_map.size());
        for (auto &v: _offset_strides_map) {
            _offset_stride_views[v.second] = &(v.first);
        }
    }
}


} // jitk
} // bohrium
