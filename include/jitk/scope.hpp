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
#include <jitk/symbol_table.hpp>

namespace bohrium {
namespace jitk {

class Scope {
public:
    const SymbolTable &symbols;
    const Scope * const parent;
private:
    std::set<const bh_base*> _tmps; // Set of temporary arrays
    std::set<const bh_base*> _scalar_replacements_rw; // Set of scalar replaced arrays that both reads and writes
    std::set<bh_view> _scalar_replacements_r; // Set of scalar replaced arrays
    std::set<InstrPtr> _omp_atomic; // Set of instructions that should be guarded by OpenMP atomic
    std::set<InstrPtr> _omp_critical; // Set of instructions that should be guarded by OpenMP critical
    std::set<bh_base*> _declared_base; // Set of bases that have been locally declared (e.g. a temporary variable)
    std::set<bh_view> _declared_view; // Set of views that have been locally declared (e.g. a temporary variable)
    std::set<bh_view, OffsetAndStrides_less> _declared_idx; // Set of indexes that have been locally declared
public:
    template<typename T1, typename T2>
    Scope(const SymbolTable &symbols,
          const Scope *parent,
          const std::set<bh_base *> &tmps,
          const T1 &scalar_replacements_rw,
          const T2 &scalar_replacements_r) : symbols(symbols), parent(parent) {
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
            assert(symbols.baseID(base) >= 0);
        }
        for (const bh_base* base: _tmps) {
            assert(_scalar_replacements_rw.find(base) == _scalar_replacements_rw.end());
            assert(symbols.baseID(base) >= 0);
        }
    #endif
    }

    // Check if 'base' is temporary
    bool isTmp(const bh_base *base) const {
        if (util::exist(_tmps, base)) {
            return true;
        } else if (parent != nullptr) {
            return parent->isTmp(base);
        } else {
            return false;
        }
    }

    // Check if 'base' has been scalar replaced read-only or read/write
    bool isScalarReplaced_R(const bh_view &view) const {
        if (util::exist(_scalar_replacements_r, view)) {
            return true;
        } else if (parent != nullptr) {
            return parent->isScalarReplaced_R(view);
        } else {
            return false;
        }
    }
    bool isScalarReplaced_RW(const bh_base *base) const {
        if (util::exist(_scalar_replacements_rw, base)) {
            return true;
        } else if (parent != nullptr) {
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

    // Insert and check if 'instr' should be guarded by OpenMP atomic
    void insertOpenmpAtomic(const InstrPtr &instr) {
        _omp_atomic.insert(instr);
    }
    bool isOpenmpAtomic(const InstrPtr &instr) const {
        if (_omp_atomic.find(instr) != _omp_atomic.end()) {
            return true;
        } else if (parent != nullptr) {
            return parent->isOpenmpAtomic(instr);
        } else {
            return false;
        }
    }

    // Insert and check if 'base' should be guarded by OpenMP critical
    void insertOpenmpCritical(const InstrPtr &instr) {
        _omp_critical.insert(instr);
    }
    bool isOpenmpCritical(const InstrPtr &instr) const {
        if (_omp_critical.find(instr) != _omp_critical.end()) {
            return true;
        } else if (parent != nullptr) {
            return parent->isOpenmpCritical(instr);
        } else {
            return false;
        }
    }

    // Check if 'view' has been locally declared (e.g. a temporary variable)
    bool isBaseDeclared(const bh_base *base) const {
        if (util::exist_nconst(_declared_base, base)) {
            return true;
        } else if (parent != nullptr) {
            return parent->isBaseDeclared(base);
        } else {
            return false;
        }
    }
    bool isViewDeclared(const bh_view &view) const {
        if (util::exist(_declared_view, view)) {
            return true;
        } else if (parent != nullptr) {
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
        } else if (parent != nullptr) {
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
