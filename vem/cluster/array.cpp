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
#include <bh.h>
#include "pgrid.h"


//Maps for translating between global and local base arrays.
static std::map<bh_base*, bh_base*> map_global2local;
static StaticStore<bh_base> local_ary_store(512);


/* Returns the local number of elements in an base array.
 *
 * @rank        The rank of the local process
 * @global_ary  The global base array
 * @return      The array size (in number of elements)
 */
bh_intp array_local_nelem(int rank, const bh_base *global_ary)
{
    bh_intp totalsize = global_ary->nelem;
    bh_intp localsize = totalsize / pgrid_worldsize;
    if(rank == pgrid_worldsize-1)//The last process gets the rest
        localsize += totalsize % pgrid_worldsize;
    return localsize;
}


/* Returns the local base array if it exist.
 *
 * @global_ary  The global base array
 * @return      The local base array or NULL
 */
bh_base* array_get_existing_local(bh_base *global_ary)
{
    return map_global2local[global_ary];
}


/* Returns the local base array.
 *
 * @global_ary  The global base array
 * @return      The local base array
 */
bh_base* array_get_local(bh_base *global_ary)
{
    bh_base *local_ary = map_global2local[global_ary];
    if(local_ary == NULL)
    {
        local_ary = local_ary_store.c_next();
        bh_create_base(global_ary->type,
                       array_local_nelem(pgrid_myrank, global_ary),
                       &local_ary);
        map_global2local[global_ary] = local_ary;
    }
    return local_ary;
}


/* Remove the local base array.
 *
 * @local_ary The local base array
 */
void array_rm_local(bh_base *local_ary)
{
    //Find the global array
    bh_base *global_ary = NULL;
    {
        std::map<bh_base*,bh_base*>::iterator it;
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



