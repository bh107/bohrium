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
#include <sys/mman.h>
#include <iostream>
#include <sstream>
#include <cstring>

#include <bh_base.hpp>

using namespace std;

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

void bh_data_malloc(bh_base *base) {
    if (base == nullptr) return;
    if (base->data != nullptr) return;

    int64_t bytes = bh_base_size(base);

    // We allow zero sized arrays.
    if (bytes == 0) return;

    if (bytes < 0) {
        throw runtime_error("Cannot allocate less than zero bytes.");
    }

    //Allocate page-size aligned memory.
    //The MAP_PRIVATE and MAP_ANONYMOUS flags is not 100% portable. See:
    //<http://stackoverflow.com/questions/4779188/how-to-use-mmap-to-allocate-a-memory-in-heap>
    base->data = mmap(0, bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (base->data == MAP_FAILED or base->data == nullptr) {
        stringstream ss;
        ss << "bh_data_malloc() could not allocate a data region. Returned error code: " << strerror(errno);
        throw runtime_error(ss.str());
    }
}

void bh_data_free(bh_base *base) {
    if (base == nullptr) return;
    if (base->data == nullptr) return;

    if (munmap(base->data, bh_base_size(base)) != 0) {
        stringstream ss;
        ss << "bh_data_free() could not free a data region. " << "Returned error code: " << strerror(errno);
        throw runtime_error(ss.str());
    }
    base->data = nullptr;
}

int64_t bh_base_size(const bh_base *base) {
    return base->nelem * bh_type_size(base->type);
}
