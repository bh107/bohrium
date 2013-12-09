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

#ifndef __BH_VEM_CLUSTER_COMM_H
#define __BH_VEM_CLUSTER_COMM_H

#include <bh.h>
#include "array.h"


/* Distribute the global base array data to all slave processes.
 * The master-process MUST have allocated the @global_ary data.
 * NB: this is a collective operation.
 *
 * @global_ary Global base array
 */
void comm_master2slaves(bh_base *global_ary);


/* Gather the global base array data at the master processes.
 * NB: this is a collective operation.
 *
 * @global_ary Global base array
 */
void comm_slaves2master(bh_base *global_ary);


/* Communicate array data such that the processes can apply local computation.
 * NB: The process that owns the data and the process where the data is located
 *     must both call this function.
 *
 * @chunk          The local array chunk to communicate
 * @sending_rank   The rank of the sending process
 * @receiving_rank The rank of the receiving process, e.g. the process that should
 *                 apply the computation
 */
void comm_array_data(const bh_view &chunk, int sending_rank, int receiving_rank);


/* Communicate array data such that the processes can apply local computation.
 * This function may reshape the input array chunk.
 * NB: The process that owns the data and the process where the data is located
 *     must both call this function.
 *
 * @chunk The local array chunk to communicate
 * @receiving_rank The rank of the receiving process, e.g. the process that should
 *                 apply the computation
 */
void comm_array_data(const ary_chunk &chunk, int receiving_rank);

#endif
