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
#include <mpi.h>
#include <assert.h>
#include "process_grid.h"

//Process prid dimension information - one for every dimension-order.
static int *pgrid_dim_strides[CPHVB_MAXDIM];
static int *pgrid_dim_sizes[CPHVB_MAXDIM];

/*===================================================================
 *
 * Initiate the MPI process grid.
 */
void pgrid_init(void)
{
    int i,j;
    int provided;
    int flag;

    //Make sure we only initialize once.
    MPI_Initialized(&flag);
    if (flag)
    {
        fprintf(stderr, "CLUSTER-VEM Warning - multiple "
                        "initialization attempts.\n");
        return;
    }

    //We make use of MPI_Init_thread even though we only ask for
    //a MPI_THREAD_SINGLE level thread-safety because MPICH2 only
    //supports MPICH_ASYNC_PROGRESS when MPI_Init_thread is used.
    //Note that when MPICH_ASYNC_PROGRESS is defined the thread-safety
    //level will automatically be set to MPI_THREAD_MULTIPLE.
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &provided);

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

    //Save the pgrid_dim_sizes and compute the pgrid_dim_strides.
    for(i=0; i<CPHVB_MAXDIM; i++)
    {
        int ndims = i+1;

        pgrid_dim_sizes[i] = malloc(ndims*sizeof(int));
        pgrid_dim_strides[i] = malloc(ndims*sizeof(int));

        memset(pgrid_dim_sizes[i], 0, ndims*sizeof(int));
        MPI_Dims_create(worldsize, ndims, pgrid_dim_sizes[i]);

        //Set cartesian information.
        memset(pgrid_dim_strides[i], 0, ndims*sizeof(int));

        //Compute strides for all dims. Using row-major like MPI.
        //A 2x2 process grid looks like:
        //    coord (0,0): rank 0.
        //    coord (0,1): rank 1.
        //    coord (1,0): rank 2.
        //    coord (1,1): rank 3.
        for(j=0; j<ndims; j++)
        {
            int stride = 1, s;
            for(s=j+1; s<ndims; s++)
                stride *= pgrid_dim_sizes[i][s];
            pgrid_dim_strides[i][j] = stride;
        }
    }
}/* pgrid_init */

/*===================================================================
 *
 * Finalize the MPI process grid.
 */
void pgrid_finalize(void)
{
    int i;

    for(i=0; i < CPHVB_MAXDIM; i++)
    {
        free(pgrid_dim_strides[i]);
        free(pgrid_dim_sizes[i]);
    }

    MPI_Finalize();

} /* pgrid_finalize */

/*===================================================================
 *
 * Computes the number of elements in a dimension of a distributed
 * array owned by the MPI-process indicated by proc_dim_rank.
 * From Fortran source: http://www.cs.umu.se/~dacke/ngssc/numroc.f
 * Private
*/
cphvb_intp pgrid_numroc(cphvb_intp nelem_in_dim, cphvb_intp block_size,
                        int proc_dim_rank, int nproc_in_dim,
                        int first_process)
{
    //Figure process's distance from source process.
    int mydist = (nproc_in_dim + proc_dim_rank - first_process) %
                  nproc_in_dim;

    //Figure the total number of whole NB blocks N is split up into.
    cphvb_intp nblocks = nelem_in_dim / block_size;

    //Figure the minimum number of elements a process can have.
    cphvb_intp numroc = nblocks / nproc_in_dim * block_size;

    //See if there are any extra blocks
    cphvb_intp extrablocks = nblocks % nproc_in_dim;

    //If I have an extra block.
    if(mydist < extrablocks)
        numroc += block_size;

    //If I have last block, it may be a partial block.
    else if(mydist == extrablocks)
        numroc += nelem_in_dim % block_size;

    return numroc;
} /* dnumroc */

/*===================================================================
 *
 * Process grid coords <-> MPI rank.
 */
int pgrid2rank(int ndims, const int coords[CPHVB_MAXDIM])
{
    int *strides = pgrid_dim_strides[ndims-1];
    int rank = 0;
    int i;
    for(i=0; i<ndims; i++)
        rank += coords[i] * strides[i];
    assert(rank < worldsize);
    return rank;
}
void rank2pgrid(int ndims, int rank, int coords[CPHVB_MAXDIM])
{
    int i;
    int *strides = pgrid_dim_strides[ndims-1];
    memset(coords, 0, ndims*sizeof(int));
    for(i=0; i<ndims; i++)
    {
        coords[i] = rank / strides[i];
        rank = rank % strides[i];
    }
} /* pgrid2rank & rank2pgrid */

/*===================================================================
 *
 * Returns the dimension sizes for a given dimension-order 'ndims'
 *
*/
int *pgrid_dim_size(int ndims)
{
    return pgrid_dim_sizes[ndims-1];
}


