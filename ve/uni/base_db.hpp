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

#ifndef __BH_VE_UNI_BASE_DB_H
#define __BH_VE_UNI_BASE_DB_H

#include <map>
#include <vector>

#include <bh_array.hpp>

/* BaseDB is a database over base arrays. The main feature is getBases(),
 * which always returns the bases in the order they where inserted.
 */
class BaseDB {
  private:
    std::map<bh_base*, size_t> _map;
    std::vector<bh_base*> _vec; // Vector of the bases where the vector index corresponds to a ID.
    std::set<bh_base*> _tmps; // Set of temporary arrays
    std::set<bh_base*> _scalar_replacements; // Set of scalar replaced arrays
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
    size_t operator[] (bh_base* base) const {
        return _map.at(base);
    }

    // Add the set of 'temps' as temporary arrays.
    // NB: the arrays should exist in this database already
    void insertTmp(const std::set<bh_base*> &temps) {
        #ifndef NDEBUG
        for (bh_base* b: temps)
            assert(_map.find(b) != _map.end());
        #endif
        _tmps.insert(temps.begin(), temps.end());
    }

    // Check if 'base' is temporary
    bool isTmp(bh_base* base) const {
        return _tmps.find(base) != _tmps.end();
    }

    // Add the 'base' as scalar replaced array
    // NB: 'base' should exist in this database already
    void insertScalarReplacement(bh_base* base) {
        assert(_map.find(base) != _map.end());
        _scalar_replacements.insert(base);
    }

    // Erase 'base' from the set of scalar replaced arrays
    void eraseScalarReplacement(bh_base* base) {
        _scalar_replacements.erase(base);
    }

    // Check if 'base' has been scalar replaced
    bool isScalarReplaced(bh_base* base) const {
        return _scalar_replacements.find(base) != _scalar_replacements.end();
    }
};

#endif
