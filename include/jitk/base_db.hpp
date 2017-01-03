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

#include <bh_array.hpp>
#include <bh_util.hpp>

namespace bohrium {
namespace jitk {

/* BaseDB is a database over base arrays. The main feature is getBases(),
 * which always returns the bases in the order they where inserted.
 */
class BaseDB {
  private:
    std::map<bh_base*, size_t> _map;
    std::vector<bh_base*> _vec; // Vector of the bases where the vector index corresponds to a ID.
    std::set<bh_base*> _tmps; // Set of temporary arrays
    std::set<bh_base*> _scalar_replacements; // Set of scalar replaced arrays
    std::set<bh_base*> _omp_atomic; // Set of arrays that should be guarded by OpenMP atomic
    std::set<bh_base*> _omp_critical; // Set of arrays that should be guarded by OpenMP critical
  public:
    BaseDB() {};

    // Insert a 'key'. Returns false and does nothing if 'key' exist, else true
    bool insert(bh_base* base) {
        if (_map.insert(std::make_pair(base, _vec.size())).second) {
            _vec.push_back(base);
            return true;
        } else {
            return false;
        }
    };

    // Return a vector of all bases in the order they where inserted
    const std::vector<bh_base*> &getBases() const {
        return _vec;
    }

    // Number of bases in the collection
    size_t size() const {
        return _vec.size();
    }

    // Get the ID of 'base', throws exception if 'base' doesn't exist
    size_t operator[] (const bh_base* base) const {
        return _map.at(const_cast<bh_base*>(base));
    }

    // Add the set of 'temps' as temporary arrays.
    // NB: the arrays should exist in this database already
    void insertTmp(const std::set<bh_base*> &temps) {
        #ifndef NDEBUG
        for (bh_base* b: temps)
            assert(util::exist(_map, b));
        #endif
        _tmps.insert(temps.begin(), temps.end());
    }

    // Check if 'base' is temporary
    bool isTmp(const bh_base* base) const {
        return util::exist(_tmps, base);
    }

    // Add the 'base' as scalar replaced array
    // NB: 'base' should exist in this database already
    void insertScalarReplacement(bh_base* base) {
        assert(util::exist(_map, base));
        _scalar_replacements.insert(base);
    }

    // Erase 'base' from the set of scalar replaced arrays
    void eraseScalarReplacement(bh_base* base) {
        _scalar_replacements.erase(base);
    }

    // Check if 'base' has been scalar replaced
    bool isScalarReplaced(const bh_base* base) const {
        return util::exist(_scalar_replacements, base);
    }

    // Insert and check if 'base' should be guarded by OpenMP atomic
    void insertOpenmpAtomic(bh_base* base) {
        _omp_atomic.insert(base);
    }
    bool isOpenmpAtomic(const bh_base* base) const {
        return util::exist(_omp_atomic, base);
    }

    // Insert and check if 'base' should be guarded by OpenMP critical
    void insertOpenmpCritical(bh_base* base) {
        _omp_critical.insert(base);
    }
    bool isOpenmpCritical(const bh_base* base) const {
        return util::exist(_omp_critical, base);
    }

};


} // jitk
} // bohrium

#endif
