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
#include "pgrid.h"

int pgrid_myrank, pgrid_worldsize;


/*===================================================================
 *
 * Initiate the MPI process grid.
 */
cphvb_error pgrid_init(void)
{
    int provided;
    int flag;

    //Make sure we only initialize once.
    MPI_Initialized(&flag);
    if (flag)
    {
        fprintf(stderr, "[CLUSTER-VEM] Warning - multiple "
                        "initialization attempts.\n");
        return CPHVB_SUCCESS;
    }

    //We make use of MPI_Init_thread even though we only ask for
    //a MPI_THREAD_SINGLE level thread-safety because MPICH2 only
    //supports MPICH_ASYNC_PROGRESS when MPI_Init_thread is used.
    //Note that when MPICH_ASYNC_PROGRESS is defined the thread-safety
    //level will automatically be set to MPI_THREAD_MULTIPLE.
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &provided);

    MPI_Comm_rank(MPI_COMM_WORLD, &pgrid_myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &pgrid_worldsize);

    printf("my rank %d of %d\n", pgrid_myrank, pgrid_worldsize);

    return CPHVB_SUCCESS;
}/* pgrid_init */

/*===================================================================
 *
 * Finalize the MPI process grid.
 */
cphvb_error pgrid_finalize(void)
{

    MPI_Finalize();

    return CPHVB_SUCCESS;
} /* pgrid_finalize */
