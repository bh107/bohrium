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
 * @return Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
bh_error bh_create_base(bh_type    type,
                        bh_index   nelements,
                        bh_base**  new_base)
{

    bh_base *base = (bh_base *) malloc(sizeof(bh_base));
    if(base == NULL) {
        return BH_OUT_OF_MEMORY;
    }

    base->type = type;
    base->nelem = nelements;
    base->data = NULL;
    *new_base = base;

    return BH_SUCCESS;
}

// Returns the label of this base array
// NB: generated a new label if necessary
static map<const bh_base*, unsigned int>label_map;
unsigned int bh_base::get_label() const
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
 * @return  Error code (BH_SUCCESS, BH_ERROR, BH_OUT_OF_MEMORY)
 */
bh_error bh_data_malloc(bh_base* base)
{
    bh_intp bytes;

    if(base == NULL) return BH_SUCCESS;
    if(base->data != NULL) return BH_SUCCESS;

    bytes = bh_base_size(base);

    // We allow zero sized arrays.
    if(bytes == 0) return BH_SUCCESS;
    if(bytes < 0) return BH_ERROR;

    base->data = bh_memory_malloc(bytes);
    if(base->data == NULL) {
        int errsv = errno; // mmap() sets the errno.
        printf("bh_data_malloc() could not allocate a data region. "
               "Returned error code: %s.\n", strerror(errsv));
        return BH_OUT_OF_MEMORY;
    }

    return BH_SUCCESS;
}

/* Frees data memory for the given view.
 * For convenience, the view is allowed to be NULL.
 *
 * @base    The base in question
 * @return  Error code (BH_SUCCESS, BH_ERROR)
 */
bh_error bh_data_free(bh_base* base)
{
    bh_intp bytes;

    if(base == NULL) return BH_SUCCESS;
    if(base->data == NULL) return BH_SUCCESS;

    bytes = bh_base_size(base);

    if(bh_memory_free(base->data, bytes) != 0) {
        int errsv = errno; // munmmap() sets the errno.
        printf("bh_data_free() could not free a data region. "
               "Returned error code: %s.\n", strerror(errsv));
        return BH_ERROR;
    }

    base->data = NULL;
    return BH_SUCCESS;
}

/* Size of the base array in bytes
 *
 * @base    The base in question
 * @return  The size of the base array in bytes
 */
bh_index bh_base_size(const bh_base *base)
{
    return base->nelem * bh_type_size(base->type);
}
