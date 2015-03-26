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

#ifndef __BH_ARRAY_H
#define __BH_ARRAY_H

#include <algorithm>
#include <stdbool.h>
#include "bh_type.h"
#include "bh_win.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BH_MAXDIM (16)

#ifndef __BH_BASE
#define __BH_BASE
struct bh_base
{
    /// Pointer to the actual data.
    bh_data_ptr   data;

    /// The type of data in the array
    bh_type       type;

    /// The number of elements in the array
    bh_index      nelem;

    bool operator<(const bh_base& other) const
    {
        if (data < other.data) return true; 
        if (other.data < data) return false; 
        if (type < other.type) return true; 
        if (other.type < type) return false; 
        if (nelem < other.nelem) return true; 
        if (other.nelem < nelem) return false;
        return false;
    }
    bool operator==(const bh_base& other) const
    {
        if (data == other.data && 
            type == other.type && 
            nelem == other.nelem) return true; 
        return false;
    }
};
#endif

struct bh_view
{
    /// Pointer to the base array.
    bh_base*      base;

    /// Index of the start element
    bh_index      start;

    /// Number of dimentions
    bh_intp       ndim;

    /// Number of elements in each dimention
    bh_index      shape[BH_MAXDIM];

    /// The stride for each dimention
    bh_index      stride[BH_MAXDIM];

    bool operator<(const bh_view& other) const
    {
        if (base < other.base) return true; 
        if (other.base < base) return false; 
        if (start < other.start) return true; 
        if (other.start < start) return false;
        if (ndim < other.ndim) return true; 
        if (other.ndim < ndim) return false; 
        for (bh_intp i = 0; i < ndim; ++i)
        {
            if (shape[i] < other.shape[i]) return true;
            if (other.shape[i] < shape[i]) return false;
        }
        for (bh_intp i = 0; i < ndim; ++i)
        {
            if (stride[i] < other.stride[i]) return true;
            if (other.stride[i] < stride[i]) return false;
        }
        return false;
    }
    bool operator==(const bh_view& other) const
    {
        if (base != other.base) return false; 
        if (ndim != other.ndim) return false; 
        if (start != other.start) return false; 
        for (bh_intp i = 0; i < ndim; ++i)
            if (shape[i] != other.shape[i]) return false;
        for (bh_intp i = 0; i < ndim; ++i)
            if (stride[i] != other.stride[i]) return false;
        return true;
    }
};

/** Create a new base array.
 *
 * @param type The type of data in the array
 * @param nelements The number of elements
 * @param new_base The handler for the newly created base
 * @return Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
DLLEXPORT bh_error bh_create_base(bh_type    type,
                                  bh_index   nelements,
                                  bh_base**  new_base);

/** Destroy the base array.
 *
 * @param base  The base array in question
 */
DLLEXPORT void bh_destroy_base(bh_base**  base);

#ifdef __cplusplus
}
#endif

#endif

