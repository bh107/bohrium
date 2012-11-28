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
#include <map>
#include <set>
#include <assert.h>
#include <StaticStore.hpp>
#include "darray.h"
#include "pgrid.h"


static std::map<cphvb_intp, cphvb_array*> map_master2slave;
static std::map<cphvb_array*,darray*> map_slave2master;
static StaticStore<darray> slave_ary_store(512);
static std::set<cphvb_array*> slave_known_arrays;


/* Insert the new array into the array store and the array maps.
 * Note that this function is only used by the slaves
 * 
 * @master_ary The master array to register locally
 * @return Pointer to the registered array.
 */
cphvb_array* darray_new_slave_array(const cphvb_array *master_ary, cphvb_intp master_id)
{
    assert(pgrid_myrank > 0);
    darray *ary = slave_ary_store.c_next();
    ary->global_ary = *master_ary;
    ary->id = master_id; 
    
    assert(map_master2slave.count(ary->id) == 0);
    assert(map_slave2master.count(&ary->global_ary) == 0);

    map_master2slave[ary->id] = &ary->global_ary;
    map_slave2master[&ary->global_ary] = ary;
    return &ary->global_ary;
}


/* Get the slave array.
 * Note that this function is only used by the slaves
 *
 * @master_array_id The master array id, which is the data pointer 
 *                  in the address space of the master-process.
 * @return Pointer to the registered array.
 */
cphvb_array* darray_master2slave(cphvb_intp master_array_id)
{
    assert(pgrid_myrank > 0);
    return map_master2slave[master_array_id];
}


/* Check if the slave array exist.
 * Note that this function is only used by the slaves
 *
 * @master_array_id The master array id, which is the data pointer 
 *                  in the address space of the master-process.
 * @return True when the slave array exist locally.
 */
bool darray_slave_exist(cphvb_intp master_array_id)
{
    assert(pgrid_myrank > 0);
    return map_master2slave.count(master_array_id) > 0;
}


/* Register the array as known by all the slaves.
 * Note that this function is only used by the master
 *
 * @ary The array that now is known.
 */
void darray_slave_known_insert(cphvb_array *ary)
{
    assert(pgrid_myrank == 0);
    slave_known_arrays.insert(ary);
}


/* Check if the array is known by all the slaves.
 * Note that this function is only used by the master
 *
 * @ary The array that should be checked.
 * @return True if the array is known by all slave-processes
 */
bool darray_slave_known_check(cphvb_array *ary)
{
    assert(pgrid_myrank == 0);
    return slave_known_arrays.count(ary) > 0;
}


/* Remove the array as known by all the slaves.
 * Note that this function is used both by the master and slaves
 *
 * @ary The array that now is unknown.
 */
void darray_slave_known_remove(cphvb_array *ary)
{
    if(pgrid_myrank == 0)
    {
        assert(darray_slave_known_check(ary));
        slave_known_arrays.erase(ary);
    }
    else
    {
        assert(map_slave2master.count(ary) == 1);
        darray *dary = map_slave2master[ary];
        map_slave2master.erase(ary);
        assert(map_master2slave.count(dary->id) == 1);
        map_master2slave.erase(dary->id);
        slave_ary_store.erase(dary);
    }
}    

