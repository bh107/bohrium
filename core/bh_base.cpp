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

#include <map>
#include <iostream>
#include <sstream>
#include <bh_base.hpp>
#include <bh_malloc_cache.hpp>

using namespace std;
using namespace bohrium;

// Returns the label of this base array
// NB: generated a new label if necessary
static map<const bh_base *, size_t> _label_map;

size_t bh_base::get_label() const {
    if (_label_map.find(this) == _label_map.end()) {
        _label_map[this] = _label_map.size();
    }
    return _label_map[this];
}

ostream &operator<<(ostream &out, const bh_base &b) {
    out << "a" << b.get_label() << "{dtype: " << bh_type_text(b.type) << ", nelem: " << b.nelem 
        << ", address: " << &b << "}";
    return out;
}

string bh_base::str() const {
    stringstream ss;
    ss << *this;
    return ss.str();
}