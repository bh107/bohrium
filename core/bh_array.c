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

#include <bh.h>

/* Create a new array.
 *
 * @base Pointer to the base array. If NULL this is a base array
 * @type The type of data in the array
 * @ndim Number of dimensions
 * @start Index of the start element (always 0 for base-array)
 * @shape[BH_MAXDIM] Number of elements in each dimention
 * @stride[BH_MAXDIM] The stride for each dimention
 * @new_array The handler for the newly created array
 * @return Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
bh_error bh_create_array(bh_array*   base,
                               bh_type     type,
                               bh_intp     ndim,
                               bh_index    start,
                               bh_index    shape[BH_MAXDIM],
                               bh_index    stride[BH_MAXDIM],
                               bh_array**  new_array)
{
    bh_array *ary = (bh_array *) malloc(sizeof(bh_array));
    if(ary == NULL)
        return BH_OUT_OF_MEMORY;

    ary->base  = base;
    ary->type  = type;
    ary->ndim  = ndim;
    ary->start = start;
    ary->data  = NULL;
    memcpy(ary->shape, shape, ndim * sizeof(bh_index));
    memcpy(ary->stride, stride, ndim * sizeof(bh_index));

#ifdef BH_TRACE
    fprintf(stderr, "Created array %lld", array);
    if (array->base != NULL)
        fprintf(stderr, " -> %lld", array->base);
    fprintf(stderr, "\n");
#endif
    
    *new_array = ary;
    return BH_SUCCESS;
}

/* Destroy array.
 *
 * @array The array to destroy
 * @return Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
bh_error bh_destroy_array(bh_array* array)
{
    free(array);
    return BH_SUCCESS;
}
