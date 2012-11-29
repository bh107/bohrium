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
#include "pgrid.h"

/* Returns the local number of elements in an array.
 * 
 * @rank The rank of the local process
 * @global_ary The global array 
 * @return The array size (in number of elements)
 */
cphvb_intp array_local_nelem(int rank, const cphvb_array *global_ary)
{
    
    cphvb_intp totalsize = cphvb_nelements(global_ary->ndim, global_ary->shape);
    cphvb_intp localsize = totalsize / pgrid_worldsize;
    if(rank == pgrid_worldsize-1)//The last process gets the rest
        localsize += totalsize % pgrid_worldsize;
    return localsize;
}

