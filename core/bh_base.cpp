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
#include <string>
#include <iostream>
#include <sstream>
#include <cstring>

#include <bh_base.hpp>
#include <bh_memory.h>

using namespace std;

/** Create a new base array.
 *
 * @param type The type of data in the array
 * @param nelements The number of elements
 * @param new_base The handler for the newly created base
 */
void bh_create_base(bh_type type, int64_t nelements, bh_base** new_base)
{
    bh_base *base = (bh_base *) malloc(sizeof(bh_base));

    if(base == NULL) {
        throw runtime_error("Out of memeory in bh_create_base()");
    }

    base->type  = type;
    base->nelem = nelements;
    base->data  = NULL;
    *new_base   = base;
}

// Returns the label of this base array
// NB: generated a new label if necessary
static map<const bh_base*, size_t>label_map;
size_t bh_base::get_label() const
{
   if(label_map.find(this) == label_map.end()) {
       label_map[this] = label_map.size();
   }

   return label_map[this];
}

ostream& operator<<(ostream& out, const bh_base& b)
{
    unsigned int label = b.get_label();
    out << "a" << label << "{dtype: " << bh_type_text(b.type) << ", nelem: " \
        << b.nelem << ", address: " << &b << "}";
    return out;
}

string bh_base::str() const
{
    stringstream ss;
    ss << *this;
    return ss.str();
}

/** Destroy the base array.
 *
 * @param base  The base array in question
 */
void bh_destroy_base(bh_base**  base)
{
    bh_base *b = *base;
    free(b);
    b = NULL;
}

/* Allocate data memory for the given base if not already allocated.
 * For convenience, the base is allowed to be NULL.
 *
 * @base    The base in question
 */
void bh_data_malloc(bh_base* base)
{
    int64_t bytes;

    if(base == NULL) return;
    if(base->data != NULL) return;

    bytes = bh_base_size(base);

    // We allow zero sized arrays.
    if(bytes == 0) return;

    if(bytes < 0) {
        throw runtime_error("Cannot allocate less than zero bytes.");
    }

    base->data = bh_memory_malloc(bytes);

    if(base->data == NULL) {
        stringstream ss;
        ss << "bh_data_malloc() could not allocate a data region. " \
           << "Returned error code: " << strerror(errno);
        throw runtime_error(ss.str());
    }
}

/* Frees data memory for the given view.
 * For convenience, the view is allowed to be NULL.
 *
 * @base    The base in question
 */
void bh_data_free(bh_base* base)
{
    int64_t bytes;

    if(base == NULL) return;
    if(base->data == NULL) return;

    bytes = bh_base_size(base);

    if(bh_memory_free(base->data, bytes) != 0) {
        stringstream ss;
        ss << "bh_data_free() could not free a data region. " \
           << "Returned error code: " << strerror(errno);
        throw runtime_error(ss.str());
    }

    base->data = NULL;
    return;
}

/* Size of the base array in bytes
 *
 * @base    The base in question
 * @return  The size of the base array in bytes
 */
int64_t bh_base_size(const bh_base *base)
{
    return base->nelem * bh_type_size(base->type);
}
