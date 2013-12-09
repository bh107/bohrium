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

#ifndef __BH_VEM_CLUSTER_TIMING_H
#define __BH_VEM_CLUSTER_TIMING_H

//Timing of events
extern bh_intp timing_comm_p2p, timing_exec_execute, timing_flush, timing_dispatch, timing_reduce, timing_mapping, timing_dispatch_array_data;

/* Initiate the timings related to the cluster-vem */
void timing_init(void);

#endif
