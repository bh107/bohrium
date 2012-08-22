/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/

#include <cphvb.h>

/* Create a new array.
 *
 * @base Pointer to the base array. If NULL this is a base array
 * @type The type of data in the array
 * @ndim Number of dimensions
 * @start Index of the start element (always 0 for base-array)
 * @shape[CPHVB_MAXDIM] Number of elements in each dimention
 * @stride[CPHVB_MAXDIM] The stride for each dimention
 * @new_array The handler for the newly created array
 * @return Error code (CPHVB_SUCCESS, CPHVB_OUT_OF_MEMORY)
 */
cphvb_error cphvb_create_array(cphvb_array*   base,
                               cphvb_type     type,
                               cphvb_intp     ndim,
                               cphvb_index    start,
                               cphvb_index    shape[CPHVB_MAXDIM],
                               cphvb_index    stride[CPHVB_MAXDIM],
                               cphvb_array**  new_array)
{
    cphvb_array *ary = (cphvb_array *) malloc(sizeof(cphvb_array));
    if(ary == NULL)
        return CPHVB_OUT_OF_MEMORY;

    ary->base  = base;
    ary->type  = type;
    ary->ndim  = ndim;
    ary->start = start;
    ary->data  = NULL;
    memcpy(ary->shape, shape, ndim * sizeof(cphvb_index));
    memcpy(ary->stride, stride, ndim * sizeof(cphvb_index));

#ifdef CPHVB_TRACE
    fprintf(stderr, "Created array %lld", array);
    if (array->base != NULL)
        fprintf(stderr, " -> %lld", array->base);
    fprintf(stderr, "\n");
#endif
    
    *new_array = ary;
    return CPHVB_SUCCESS;
}

/* Destroy array.
 *
 * @array The array to destroy
 * @return Error code (CPHVB_SUCCESS, CPHVB_OUT_OF_MEMORY)
 */
cphvb_error cphvb_destroy_array(cphvb_array* array)
{
    free(array);
    return CPHVB_SUCCESS;
}
