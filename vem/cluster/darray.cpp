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
#include <assert.h>
#include <StaticStore.hpp>
#include "darray.h"


static std::map<cphvb_intp, cphvb_array*> map_master2slave;
static std::map<cphvb_array*,darray*> map_slave2master;
static StaticStore<darray> slave_ary_store(512);


/* Insert the new array into the array store and the array maps.
 * 
 * @master_ary The master array to register locally
 * @return Pointer to the registered array.
 */
darray* darray_new_slave_array(const darray *master_ary)
{
    darray *ary = slave_ary_store.c_next();
    *ary = *master_ary;
    assert(map_master2slave.count(ary->id) == 0);
    assert(map_slave2master.count(&ary->global_ary) == 0);

    map_master2slave[ary->id] = &ary->global_ary;
    map_slave2master[&ary->global_ary] = ary;
    return ary;
}


/* Get the slave array.
 *
 * @master_array_id The master array id, which is the data pointer 
 *                  in the address space of the master-process.
 * @return Pointer to the registered array.
 */
cphvb_array* darray_master2slave(cphvb_intp master_array_id)
{
    return map_master2slave[master_array_id];
}


/* Check if the slave array exist.
 *
 * @master_array_id The master array id, which is the data pointer 
 *                  in the address space of the master-process.
 * @return True when the slave array exist locally.
 */
bool darray_slave_exist(cphvb_intp master_array_id)
{
    return map_master2slave.count(master_array_id) > 0;
}
