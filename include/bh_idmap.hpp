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

#ifndef __BH_IDMAP_H
#define __BH_IDMAP_H

#include <map>
#include <vector>

/* IdMap is a map of keys to IDs. The main feature is getKeys(),
 * which always returns the keys in the order they where inserted.
 */
template <typename T>
class IdMap {
  private:
    std::map<T, size_t> _map;
    std::vector<T> _vec; // Vector of the keys where the vector index corresponds to a ID.
  public:
    IdMap() {};

    // Insert a 'key'. Returns false and does nothing if 'key' exist, else true
    bool insert(T key) {
        if (_map.insert(std::make_pair(key, _vec.size())).second) {
            _vec.push_back(key);
            return true;
        } else {
            return false;
        }
    };

    // Return a vector of all keys in the irder they where inserted
    const std::vector<T> &getKeys() const {
        return _vec;
    }

    // Number of keys/IDs in the collection
    size_t size() const {
        return _vec.size();
    }

    // Get the ID of 'key', throws exception if 'key' doesn't exist
    size_t operator[] (T key) const {
        return _map.at(key);
    }
};


#endif
