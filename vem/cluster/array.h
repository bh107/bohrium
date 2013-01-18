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
    bh_array *ary;
    //Local array chunk coordinate (in array elements)
    bh_intp coord[BH_MAXDIM];
}ary_chunk;
 

/* Returns the local array based on the global array if it exist.
 * NB: this function only accept base-arrays. 
 *
 * @global_ary The global array 
 * @return The local array or NULL
 */
bh_array* array_get_existing_local(bh_array *global_ary);


/* Returns the local number of elements in an array.
 * 
 * @rank The rank of the local process
 * @global_ary The global array 
 * @return The array size (in number of elements)
 */
bh_intp array_local_nelem(int rank, const bh_array *global_ary);


/* Returns the local array based on the global array.
 * NB: this function only accept base-arrays. 
 *
 * @global_ary The global array 
 * @return The local array
 */
bh_array* array_get_local(bh_array *global_ary);


/* Remove the local array based on the global array.
 * NB: this function only accept base-arrays. 
 *
 * @global_ary The global array 
 */
void array_rm_local(bh_array *global_ary);

#endif
