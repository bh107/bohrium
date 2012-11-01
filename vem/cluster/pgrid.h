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

#ifndef CLUSTER_VEM_PGRID_H
#define CLUSTER_VEM_PGRID_H
#include <mpi.h>
#include <cphvb.h>

extern int pgrid_myrank, pgrid_worldsize;

/*===================================================================
 *
 * Initiate the MPI process grid.
 * NB: must be called before the use of myrank and worldsize
 */
cphvb_error pgrid_init(void);


/*===================================================================
 *
 * Finalize the MPI process grid.
 */
cphvb_error pgrid_finalize(void);


#endif
