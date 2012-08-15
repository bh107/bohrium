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

#ifndef PROCESS_GRID_H
#define PROCESS_GRID_H
#include <mpi.h>
#include <cphvb.h>

#ifdef __cplusplus
extern "C" {
#endif

int myrank, worldsize;

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

/*===================================================================
 *
 * Returns the dimension sizes for a given dimension-order 'ndims'
 *
*/
int *pgrid_dim_size(int ndims);

/*===================================================================
 *
 * Computes the number of elements in a dimension of a distributed
 * array owned by the MPI-process indicated by proc_dim_rank.
 * From Fortran source: http://www.cs.umu.se/~dacke/ngssc/numroc.f
*/
cphvb_intp pgrid_numroc(cphvb_intp nelem_in_dim, cphvb_intp block_size,
                        int proc_dim_rank, int nproc_in_dim,
                        int first_process);

/*===================================================================
 *
 * Process grid coords <-> MPI rank.
 */
int pgrid2rank(int ndims, const int coords[CPHVB_MAXDIM]);
void rank2pgrid(int ndims, int rank, int coords[CPHVB_MAXDIM]);

#ifdef __cplusplus
}
#endif

#endif
