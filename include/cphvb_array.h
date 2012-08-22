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

#ifndef __CPHVB_ARRAY_H
#define __CPHVB_ARRAY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "cphvb_type.h"
#include "cphvb_win.h"

#define CPHVB_MAXDIM (16)

typedef struct cphvb_array cphvb_array;
struct cphvb_array
{
    //Pointer to the base array. If NULL this is a base array
    cphvb_array*     base;

    //The type of data in the array
    cphvb_type       type;

    //Number of dimentions
    cphvb_intp       ndim;

    //Index of the start element (always 0 for base-array)
    cphvb_index      start;

    //Number of elements in each dimention
    cphvb_index      shape[CPHVB_MAXDIM];

    //The stride for each dimention
    cphvb_index      stride[CPHVB_MAXDIM];

    //Pointer to the actual data. Ignored for views
    cphvb_data_ptr   data;
};

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
DLLEXPORT cphvb_error cphvb_create_array(cphvb_array*   base,
                               cphvb_type     type,
                               cphvb_intp     ndim,
                               cphvb_index    start,
                               cphvb_index    shape[CPHVB_MAXDIM],
                               cphvb_index    stride[CPHVB_MAXDIM],
                               cphvb_array**  new_array);

/* Destroy array.
 *
 * @array The array to destroy
 * @return Error code (CPHVB_SUCCESS, CPHVB_OUT_OF_MEMORY)
 */
DLLEXPORT cphvb_error cphvb_destroy_array(cphvb_array* array);


#ifdef __cplusplus
}
#endif

#endif
