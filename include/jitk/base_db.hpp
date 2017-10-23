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

#ifndef __BH_JITK_BASE_DB_H
#define __BH_JITK_BASE_DB_H

#include <map>
#include <vector>
#include <string>
#include <sstream>

#include <bh_view.hpp>
#include <bh_util.hpp>
#include <bh_config_parser.hpp>

#include <jitk/block.hpp>

namespace bohrium {
namespace jitk {

// Compare class for the index sets and maps
struct idx_less {
    // This compare is the same as view compare ('v1 < v2') but ignoring their bases
    bool operator() (const bh_view& v1, const bh_view& v2) const {
        if (v1.start < v2.start) return true;
        if (v2.start < v1.start) return false;
        if (v1.ndim < v2.ndim) return true;
        if (v2.ndim < v1.ndim) return false;
        for (int64_t i = 0; i < v1.ndim; ++i) {
            if (v1.shape[i] < v2.shape[i]) return true;
            if (v2.shape[i] < v1.shape[i]) return false;
        }
        for (int64_t i = 0; i < v1.ndim; ++i) {
            if (v1.stride[i] < v2.stride[i]) return true;
            if (v2.stride[i] < v1.stride[i]) return false;
        }
        return false;
    }
};

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
    std::map<bh_view, size_t, idx_less> _idx_map; // Mapping a index (of an array) to its ID
    std::map<bh_view, size_t, OffsetAndStrides_less> _offset_strides_map; // Mapping a offset-and-strides to its ID
    std::vector<const bh_view*> _offset_stride_views; // Vector of all offset-and-stride views
    std::set<InstrPtr, Constant_less> _constant_set; // Set of instructions to a constant ID (Order by `origin_id`)
    std::set<const bh_base*> _array_always; // Set of base arrays that should always be arrays
    std::vector<bh_base*> _params; // Vector of non-temporary arrays, which are the in-/out-puts of the JIT kernel
    std::set<bh_base*> _frees; // Set of freed arrays
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

    SymbolTable(const std::vector<InstrPtr> &instr_list, const std::set<bh_base *> &non_temp_arrays,
                bool strides_as_var, bool index_as_var, bool const_as_var,
                bool use_volatile) : _useRandom(false), use_volatile(use_volatile), strides_as_var(strides_as_var),
                                     index_as_var(index_as_var), const_as_var(const_as_var) {

        // NB: by assigning the IDs in the order they appear in the 'instr_list',
        //     the kernels can better be reused
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
                if (instr->has_constant() and not bh_opcode_is_sweep(instr->opcode)) {
                    _constant_set.insert(instr);
                }
            }
            if (instr->opcode == BH_GATHER) {
                if (not bh_is_constant(&instr->operand[1])) {
                    _array_always.insert(instr->operand[1].base);
                }
            } else if (instr->opcode == BH_SCATTER or instr->opcode == BH_COND_SCATTER) {
                _array_always.insert(instr->operand[0].base);
            } else if (instr->opcode == BH_RANDOM) {
                _useRandom = true;
            } else if (instr->opcode == BH_FREE) {
                _frees.insert(instr->operand[0].base);
            }
            // Find bases that are the parameters to the JIT kernel, which are non-temporary arrays not
            // already in `_params`. NB: the order of `_params` matches the order of the array IDs
            for(const bh_view &v: instr->operand) {
                if (not bh_is_constant(&v) and util::exist(non_temp_arrays, v.base)) {
                    if (not util::exist_linearly(_params, v.base)) {
                        _params.push_back(v.base);
                    }
                }
            }
        }
        if (strides_as_var) {
            _offset_stride_views.resize(_offset_strides_map.size());
            for(auto &v: _offset_strides_map) {
                _offset_stride_views[v.second] = &v.first;
            }
        }
    };
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
        return util::exist(_array_always, base);
    }
    // Return non-temporary arrays, which are the in-/out-puts of the JIT kernel, in the order of their IDs
    const std::vector<bh_base*> &getParams() const {
        return _params;
    }
    // Return the freed arrays
    const std::set<bh_base*> &getFrees() const {
        return _frees;
    }
    // Is any instructions use the random library?
    bool useRandom() const {
        return _useRandom;
    }
};

class Scope {
public:
    const SymbolTable &symbols;
    const Scope * const parent;
private:
    std::set<const bh_base*> _tmps; // Set of temporary arrays
    std::set<const bh_base*> _scalar_replacements_rw; // Set of scalar replaced arrays that both reads and writes
    std::set<bh_view> _scalar_replacements_r; // Set of scalar replaced arrays
    std::set<bh_view> _omp_atomic; // Set of arrays that should be guarded by OpenMP atomic
    std::set<bh_view> _omp_critical; // Set of arrays that should be guarded by OpenMP critical
    std::set<bh_base*> _declared_base; // Set of bases that have been locally declared (e.g. a temporary variable)
    std::set<bh_view> _declared_view; // Set of views that have been locally declared (e.g. a temporary variable)
    std::set<bh_view, idx_less> _declared_idx; // Set of indexes that have been locally declared
public:
    template<typename T1, typename T2>
    Scope(const SymbolTable &symbols,
          const Scope *parent,
          const std::set<bh_base *> &tmps,
          const T1 &scalar_replacements_rw,
          const T2 &scalar_replacements_r,
          const ConfigParser &config) : symbols(symbols), parent(parent) {
        for(const bh_base* base: tmps) {
            if (not symbols.isAlwaysArray(base))
                _tmps.insert(base);
        }
        for(const bh_view* view: scalar_replacements_rw) {
            if (not symbols.isAlwaysArray(view->base))
                _scalar_replacements_rw.insert(view->base);
        }
        for(const bh_view* view: scalar_replacements_r) {
            if (not symbols.isAlwaysArray(view->base))
                _scalar_replacements_r.insert(*view);
        }

        // No overlap between '_tmps', '_scalar_replacements_rw', and '_scalar_replacements_r' is allowed
    #ifndef NDEBUG
        for (const bh_view &view: _scalar_replacements_r) {
            assert(_tmps.find(view.base) == _tmps.end());
            assert(_scalar_replacements_rw.find(view.base) == _scalar_replacements_rw.end());
            assert(symbols.viewID(view) >= 0);
        }
        for (const bh_base* base: _scalar_replacements_rw) {
            assert(_tmps.find(base) == _tmps.end());
//            assert(_scalar_replacements_r.find(view) == _scalar_replacements_r.end());
            assert(symbols.baseID(base) >= 0);
        }
        for (const bh_base* base: _tmps) {
//            assert(_scalar_replacements_r.find(base) == _scalar_replacements_r.end());
            assert(_scalar_replacements_rw.find(base) == _scalar_replacements_rw.end());
            assert(symbols.baseID(base) >= 0);
        }
    #endif
    }

    // Check if 'base' is temporary
    bool isTmp(const bh_base *base) const {
        if (util::exist(_tmps, base)) {
            return true;
        } else if (parent != NULL) {
            return parent->isTmp(base);
        } else {
            return false;
        }
    }

    // Check if 'base' has been scalar replaced read-only or read/write
    bool isScalarReplaced_R(const bh_view &view) const {
        if (util::exist(_scalar_replacements_r, view)) {
            return true;
        } else if (parent != NULL) {
            return parent->isScalarReplaced_R(view);
        } else {
            return false;
        }
    }
    bool isScalarReplaced_RW(const bh_base *base) const {
        if (util::exist(_scalar_replacements_rw, base)) {
            return true;
        } else if (parent != NULL) {
            return parent->isScalarReplaced_RW(base);
        } else {
            return false;
        }
    }

    // Check if 'view' has been scalar replaced
    bool isScalarReplaced(const bh_view &view) const {
        return isScalarReplaced_R(view) or isScalarReplaced_RW(view.base);
    }

    // Check if 'view' is a regular array (not temporary, scalar-replaced etc.)
    bool isArray(const bh_view &view) const {
        return not (isTmp(view.base) or isScalarReplaced(view));
    }

    // Insert and check if 'base' should be guarded by OpenMP atomic
    void insertOpenmpAtomic(const bh_view &view) {
        _omp_atomic.insert(view);
    }
    bool isOpenmpAtomic(const bh_view &view) const {
        if (_omp_atomic.find(view) != _omp_atomic.end()) {
            return true;
        } else if (parent != NULL) {
            return parent->isOpenmpAtomic(view);
        } else {
            return false;
        }
    }

    // Insert and check if 'base' should be guarded by OpenMP critical
    void insertOpenmpCritical(const bh_view &view) {
        _omp_critical.insert(view);
    }
    bool isOpenmpCritical(const bh_view &view) const {
        if (_omp_critical.find(view) != _omp_critical.end()) {
            return true;
        } else if (parent != NULL) {
            return parent->isOpenmpCritical(view);
        } else {
            return false;
        }
    }

    // Check if 'view' has been locally declared (e.g. a temporary variable)
    bool isBaseDeclared(const bh_base *base) const {
        if (util::exist_nconst(_declared_base, base)) {
            return true;
        } else if (parent != NULL) {
            return parent->isBaseDeclared(base);
        } else {
            return false;
        }
    }
    bool isViewDeclared(const bh_view &view) const {
        if (util::exist(_declared_view, view)) {
            return true;
        } else if (parent != NULL) {
            return parent->isViewDeclared(view);
        } else {
            return false;
        }
    }
    bool isDeclared(const bh_view &view) const {
        return isBaseDeclared(view.base) or isViewDeclared(view);
    }

    // Check if 'index' has been locally declared
    bool isIdxDeclared(const bh_view &index) const {
        if (util::exist(_declared_idx, index)) {
            return true;
        } else if (parent != NULL) {
            return parent->isIdxDeclared(index);
        } else {
            return false;
        }
    }

    // Get the name (symbol) of the 'base'
    template <typename T>
    void getName(const bh_view &view, T &out) const {
        if (isTmp(view.base)) {
            out << "t" << symbols.baseID(view.base);
        } else if (isScalarReplaced(view)) {
            out << "s" << symbols.baseID(view.base);
            if (isScalarReplaced_R(view)) {
                out << "_" << symbols.viewID(view);;
            }
        } else {
            out << "a" << symbols.baseID(view.base);
        }
    }
    std::string getName(const bh_view &view) const {
        std::stringstream ss;
        getName(view, ss);
        return ss.str();
    }

    // Write the variable declaration of 'base' using 'type_str' as the type string
    template <typename T>
    void writeDeclaration(const bh_view &view, const std::string &type_str, T &out) {
        assert(not isDeclared(view));

        if (symbols.use_volatile) {
            out << "volatile ";
        }

        out << type_str << " " << getName(view) << ";";

        if (isTmp(view.base) or isScalarReplaced_RW(view.base)) {
            _declared_base.insert(view.base);
        } else if (isScalarReplaced_R(view)){
            _declared_view.insert(view);
        } else {
            throw std::runtime_error("calling writeDeclaration() on a regular array");
        }
    }

    // Get the name (symbol) of the 'base'
    template <typename T>
    void getIdxName(const bh_view &view, T &out) const {
        out << "idx" << symbols.idxID(view);
    }
    std::string getIdxName(const bh_view &view) const {
        std::stringstream ss;
        getIdxName(view, ss);
        return ss.str();
    }

    // Write the variable declaration of the index calculation of 'view' using 'type_str' as the type string
    void writeIdxDeclaration(const bh_view &view, const std::string &type_str, std::stringstream &out);
};


} // jitk
} // bohrium

#endif
