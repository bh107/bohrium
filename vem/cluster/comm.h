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
#include "array.h"

#ifndef __CPHVB_VEM_CLUSTER_COMM_H
#define __CPHVB_VEM_CLUSTER_COMM_H


/* Distribute the global array data to all slave processes.
 * NB: this is a collective operation.
 * 
 * @global_ary Global base array
 */
cphvb_error comm_master2slaves(cphvb_array *global_ary);


/* Gather the global array data at the master processes.
 * NB: this is a collective operation.
 * 
 * @global_ary Global base array
 */
cphvb_error comm_slaves2master(cphvb_array *global_ary);


/* Communicate array data such that the processes can apply local computation.
 * This function may reshape the input array.
 * NB: The process that owns the data and the process where the data is located
 *     must both call this function.
 *     
 * @local_ary The local array to communicate
 * @local_ary_ext The local array extention
 * @receiving_rank The rank of the receiving process, e.g. the process that should
 *                 apply the computation
 */
cphvb_error comm_array_data(cphvb_array *local_ary, array_ext *local_ary_ext, 
                            int receiving_rank);


#endif
