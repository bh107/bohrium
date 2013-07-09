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

#ifndef __BH_VEM_CLUSTER_DARRAY_H
#define __BH_VEM_CLUSTER_DARRAY_H


//Local array chunk
typedef struct
{
    //Process rank that owns this array chunk.
    int rank;
    //Local array chunk
    bh_view ary;
    //Global coordinate of the array chunk (in elements)
    bh_intp coord[BH_MAXDIM];
    //True when the array chunk is only temporary
    bool temporary;
}ary_chunk;

/* Returns the local number of elements in an base array.
 *
 * @rank        The rank of the local process
 * @global_ary  The global base array
 * @return      The array size (in number of elements)
 */
bh_intp array_local_nelem(int rank, const bh_base *global_ary);

/* Returns the local base array if it exist.
 *
 * @global_ary  The global base array
 * @return      The local base array or NULL
 */
bh_base* array_get_existing_local(bh_base *global_ary);

/* Returns the local base array.
 *
 * @global_ary  The global base array
 * @return      The local base array
 */
bh_base* array_get_local(bh_base *global_ary);

/* Remove the local base array.
 *
 * @local_ary The local base array
 */
void array_rm_local(bh_base *local_ary);

#endif
