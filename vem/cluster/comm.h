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

#include <cphvb.h>
#include "array.h"

#ifndef __CPHVB_VEM_CLUSTER_COMM_H
#define __CPHVB_VEM_CLUSTER_COMM_H


/* Gather or scatter the global array processes.
 * NB: this is a collective operation.
 * 
 * @scatter If true we scatter else we gather
 * @global_ary Global base array
 */
void comm_gather_scatter(int scatter, cphvb_array *global_ary);


/* Distribute the global array data to all slave processes.
 * The master-process MUST have allocated the @global_ary data.
 * NB: this is a collective operation.
 * 
 * @global_ary Global base array
 */
void comm_master2slaves(cphvb_array *global_ary);


/* Gather the global array data at the master processes.
 * NB: this is a collective operation.
 * 
 * @global_ary Global base array
 */
void comm_slaves2master(cphvb_array *global_ary);


/* Communicate array data such that the processes can apply local computation.
 * This function may reshape the input array chunk.
 * NB: The process that owns the data and the process where the data is located
 *     must both call this function.
 *     
 * @chunk The local array chunk to communicate
 * @receiving_rank The rank of the receiving process, e.g. the process that should
 *                 apply the computation
 */
void comm_array_data(ary_chunk *chunk, int receiving_rank);


#endif
