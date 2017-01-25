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

#ifndef CLUSTER_VEM_PGRID_H
#define CLUSTER_VEM_PGRID_H
#include <mpi.h>

extern int pgrid_myrank, pgrid_worldsize;

/*===================================================================
 *
 * Initiate the MPI process grid.
 * NB: must be called before the use of myrank and worldsize
 */
void pgrid_init(void);


/*===================================================================
 *
 * Finalize the MPI process grid.
 */
void pgrid_finalize(void);


#endif
