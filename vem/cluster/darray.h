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

#ifndef __CPHVB_VEM_CLUSTER_DARRAY_H
#define __CPHVB_VEM_CLUSTER_DARRAY_H


//Extension to the cphvb_array for cluster information
typedef struct
{
    //Process rank that owns the array.
    int rank;
}darray_ext;


//Extension to the cphvb_array for cluster information
typedef struct
{
    //The id of the array. This is identical with the array-struct address 
    //on the master-process.
    cphvb_intp id;
    //The global array-struct.
    cphvb_array global_ary;
}darray;


/* Insert the new array into the array store and the array maps.
 * 
 * @master_ary The master array to register locally
 * @return Pointer to the registered array.
 */
darray* darray_new_slave_array(const darray *master_ary);


/* Get the slave array.
 *
 * @master_array_id The master array id, which is the data pointer 
 *                  in the address space of the master-process.
 * @return Pointer to the registered array.
 */
cphvb_array* darray_master2slave(cphvb_intp master_array_id);


/* Check if the slave array exist.
 *
 * @master_array_id The master array id, which is the data pointer 
 *                  in the address space of the master-process.
 * @return True when the slave array exist locally.
 */
bool darray_slave_exist(cphvb_intp master_array_id);


/* Register the array as known by all the slaves.
 * Note that this function is only used by the master
 *
 * @ary The array that now is known.
 */
void darray_slave_known_insert(cphvb_array *ary);


/* Check if the array is known by all the slaves.
 * Note that this function is only used by the master
 *
 * @ary The array that should be checked.
 * @return True if the array is known by all slave-processes
 */
bool darray_slave_known_check(cphvb_array *ary);


/* Remove the array as known by all the slaves.
 * Note that this function is only used by the master
 *
 * @ary The array that now is unknown.
 */
void darray_slave_known_remove(cphvb_array *ary);




#endif
