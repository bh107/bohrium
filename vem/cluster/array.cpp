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

#include <cassert>
#include <map>
#include <StaticStore.hpp>
#include <cphvb.h>
#include "pgrid.h"


//Maps for translating between global and local arrays.
static std::map<cphvb_array*, cphvb_array*> map_global2local;
//static std::map<cphvb_array*, cphvb_array*> map_local2global;
static StaticStore<cphvb_array> local_ary_store(512);


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


/* Returns the local array based on the global array if it exist.
 * NB: this function only accept base-arrays. 
 *
 * @global_ary The global array 
 * @return The local array or NULL
 */
cphvb_array* array_get_existing_local(cphvb_array *global_ary)
{
    return map_global2local[global_ary];
}


/* Returns the local array based on the global array.
 * NB: this function only accept base-arrays. 
 *
 * @global_ary The global array 
 * @return The local array
 */
cphvb_array* array_get_local(cphvb_array *global_ary)
{
    assert(global_ary->base == NULL);
    cphvb_array *local_ary = map_global2local[global_ary];
    if(local_ary == NULL)
    {
        local_ary = local_ary_store.c_next();
        local_ary->base = NULL;
        local_ary->type = global_ary->type;
        local_ary->ndim = 1;//We always use a flatten base array.
        local_ary->start = 0;
        local_ary->shape[0] = array_local_nelem(pgrid_myrank, global_ary); 
        local_ary->stride[0] = 1;
        local_ary->data = NULL;
        map_global2local[global_ary] = local_ary;
    }
    return local_ary;
}


/* Remove the local array.
 * NB: this function only accept base-arrays. 
 *
 * @local_ary The local array 
 */
void array_rm_local(cphvb_array *local_ary)
{
    assert(local_ary->base == NULL);

    //find the global array
    cphvb_array *global_ary = NULL;
    {
        std::map<cphvb_array*,cphvb_array*>::iterator it;
        for(it=map_global2local.begin(); it != map_global2local.end(); ++it)
            if(it->second == local_ary)
            {
                assert(global_ary == NULL);
                global_ary = it->first;
            }
    }
    //If 'local_ary' is not created through array_get_local()
    //it is not in map_global2local and we will do nothing
    if(global_ary != NULL)
    {
        map_global2local.erase(global_ary);
        local_ary_store.erase(local_ary);
    }
}
   
    

