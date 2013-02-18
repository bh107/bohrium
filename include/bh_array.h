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

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "bh_type.h"
#include "bh_win.h"

#define BH_MAXDIM (16)

typedef struct bh_array bh_array;
struct bh_array
{
    /// Pointer to the base array. If NULL this is a base array
    bh_array*     base;

    /// The type of data in the array
    bh_type       type;

    /// Number of dimentions
    bh_intp       ndim;

    /// Index of the start element (always 0 for base-array)
    bh_index      start;

    /// Number of elements in each dimention
    bh_index      shape[BH_MAXDIM];

    /// The stride for each dimention
    bh_index      stride[BH_MAXDIM];

    /// Pointer to the actual data. Ignored for views
    bh_data_ptr   data;
};

/** Create a new array.
 *
 * @param base Pointer to the base array. If NULL this is a base array
 * @param type The type of data in the array
 * @param ndim Number of dimensions
 * @param start Index of the start element (always 0 for base-array)
 * @param shape[BH_MAXDIM] Number of elements in each dimention
 * @param stride[BH_MAXDIM] The stride for each dimention
 * @param new_array The handler for the newly created array
 * @return Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
DLLEXPORT bh_error bh_create_array(bh_array*   base,
                               bh_type     type,
                               bh_intp     ndim,
                               bh_index    start,
                               bh_index    shape[BH_MAXDIM],
                               bh_index    stride[BH_MAXDIM],
                               bh_array**  new_array);

/** Destroy array.
 *
 * @param array The array to destroy
 * @return Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
DLLEXPORT bh_error bh_destroy_array(bh_array* array);


#ifdef __cplusplus
}
#endif

#endif
