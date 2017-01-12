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

#include <bh_array.hpp>
#include <bh_util.hpp>

#include <jitk/kernel.hpp>

namespace bohrium {
namespace jitk {


class SymbolTable {
private:
    std::map<bh_base*, size_t> _base_map; // Mapping a base to its ID
    std::map<bh_view, size_t> _view_map; // Mapping a base to its ID

public:
    SymbolTable(const std::vector<InstrPtr> &instr_list) {
        // NB: by assigning the IDs in the order they appear in the 'instr_list',
        //     the kernels can better be reused
        for (const InstrPtr instr: instr_list) {
            for (const bh_view *view: instr->get_views()) {
                _base_map.insert(std::make_pair(view->base, _base_map.size()));
                _view_map.insert(std::make_pair(*view, _view_map.size()));
            }
        }
    };

    // Get the ID of 'base', throws exception if 'base' doesn't exist
    size_t baseID(bh_base *base) const {
        return _base_map.at(base);
    }
    // Get the ID of 'view', throws exception if 'view' doesn't exist
    size_t viewID(const bh_view &view) const {
        //return _view_map.at(view);
        return _base_map.at(view.base);
    }
};


class Scope {
public:
    const SymbolTable &symbols;
    const Scope * const parent;

private:
    std::set<bh_base*> _tmps; // Set of temporary arrays
    std::set<bh_base*> _scalar_replacements_rw; // Set of scalar replaced arrays that both reads and writes
    std::set<bh_view> _scalar_replacements_r; // Set of scalar replaced arrays
    std::set<bh_view> _omp_atomic; // Set of arrays that should be guarded by OpenMP atomic
    std::set<bh_view> _omp_critical; // Set of arrays that should be guarded by OpenMP critical
    std::set<bh_base*> _local_declared; // Set of arrays that have been locally declared (e.g. a temporary variable)
public:
    template<typename T1, typename T2>
    Scope(const SymbolTable &symbols,
          const Scope *parent,
          const std::set<bh_base *> &tmps,
          const T1 &scalar_replacements_rw,
          const T2 &scalar_replacements_r) : symbols(symbols),
                                             parent(parent),
                                             _tmps(tmps) {
        for(const bh_view* view: scalar_replacements_rw) {
            _scalar_replacements_rw.insert(view->base);
        }
        for(const bh_view* view: scalar_replacements_r) {
            _scalar_replacements_r.insert(*view);
        }

        // No overlap between '_tmps', '_scalar_replacements_rw', and '_scalar_replacements_r' is allowed
    #ifndef NDEBUG
        for (const bh_view &view: _scalar_replacements_r) {
            assert(_tmps.find(view.base) == _tmps.end());
            assert(_scalar_replacements_rw.find(view.base) == _scalar_replacements_rw.end());
        }
        for (bh_base* base: _scalar_replacements_rw) {
            assert(_tmps.find(base) == _tmps.end());
//            assert(_scalar_replacements_r.find(view) == _scalar_replacements_r.end());
        }
        for (bh_base* base: _tmps) {
//            assert(_scalar_replacements_r.find(base) == _scalar_replacements_r.end());
            assert(_scalar_replacements_rw.find(base) == _scalar_replacements_rw.end());
        }
    #endif
    }

    // Check if 'base' is temporary
    bool isTmp(const bh_base *base) const {
        if (util::exist_nconst(_tmps, base)) {
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
    bool isScalarReplaced_RW(const bh_view &view) const {
        if (util::exist(_scalar_replacements_rw, view.base)) {
            return true;
        } else if (parent != NULL) {
            return parent->isScalarReplaced_RW(view);
        } else {
            return false;
        }
    }

    // Check if 'view' has been scalar replaced
    bool isScalarReplaced(const bh_view &view) const {
        return isScalarReplaced_R(view) or isScalarReplaced_RW(view);
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

    // Insert and check if 'base' has been locally declared (e.g. a temporary variable)
    bool isDeclared(const bh_view &view) const {
        if (_local_declared.find(view.base) != _local_declared.end()) {
            return true;
        } else if (parent != NULL) {
            return parent->isDeclared(view);
        } else {
            return false;
        }
    }
    void insertDeclared(const bh_view &view) {
        assert(not isDeclared(view));
        _local_declared.insert(view.base);
    }

    // Get the name (symbol) of the 'base'
    template <typename T>
    void getName(const bh_view &view, T &out) const {
        if (isTmp(view.base)) {
            out << "t";
            out << symbols.baseID(view.base);
        } else if (isScalarReplaced(view)) {
            out << "s";
            out << symbols.baseID(view.base);
        } else {
            out << "a";
            out << symbols.baseID(view.base);
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
        out << type_str << " ";
        getName(view, out);
        out << ";";
        insertDeclared(view);
    }
};


} // jitk
} // bohrium

#endif
