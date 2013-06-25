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

/** Create a new base array.
 *
 * @param type The type of data in the array
 * @param nelements The number of elements
 * @param new_base The handler for the newly created base
 * @return Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
DLLEXPORT bh_error bh_create_base(bh_type    type,
                                  bh_index   nelements,
                                  bh_base**  new_base)
{

    bh_base *base = (bh_base *) malloc(sizeof(bh_base));
    if(base == NULL)
        return BH_OUT_OF_MEMORY;
    base->type = type;
    base->nelem = nelements;
    base->data = NULL;
    *new_base = base;

    return BH_SUCCESS:
}


/* Create a new array view.
 *
 * @base Pointer to the base array
 * @ndim Number of dimensions
 * @start Index of the start element
 * @shape[BH_MAXDIM] Number of elements in each dimention
 * @stride[BH_MAXDIM] The stride for each dimention
 * @new_view The handler for the newly created view
 * @return Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
bh_error bh_create_view(bh_base*   base,
                        bh_intp    ndim,
                        bh_index   start,
                        bh_index   shape[BH_MAXDIM],
                        bh_index   stride[BH_MAXDIM],
                        bh_view**  new_view)
{
    bh_view *view = (bh_view *) malloc(sizeof(bh_view));
    if(view == NULL)
        return BH_OUT_OF_MEMORY;

    view->base  = base;
    view->ndim  = ndim;
    view->start = start;
    memcpy(view->shape, shape, ndim * sizeof(bh_index));
    memcpy(view->stride, stride, ndim * sizeof(bh_index));

    *new_array = ary;
    return BH_SUCCESS;
}

/** Destroy array view.
 *
 * @param view The view to destroy
 * @return Error code (BH_SUCCESS, BH_OUT_OF_MEMORY)
 */
DLLEXPORT bh_error bh_destroy_array(bh_view* view);
