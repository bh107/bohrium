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
#include <mpi.h>
#include <assert.h>
#include "pgrid.h"
#include "except.h"

#include <sched.h>
#include <unistd.h>

int pgrid_myrank, pgrid_worldsize;


/*===================================================================
 *
 * Initiate the MPI process grid.
 */
void pgrid_init(void)
{
    int provided;
    int flag;
    int e;

    //Make sure we only initialize once.
    MPI_Initialized(&flag);
    if (flag)
    {
        fprintf(stderr, "[CLUSTER-VEM] Warning - multiple "
                        "initialization attempts.\n");
        return;
    }

    //We make use of MPI_Init_thread even though we only ask for
    //a MPI_THREAD_SINGLE level thread-safety because MPICH2 only
    //supports MPICH_ASYNC_PROGRESS when MPI_Init_thread is used.
    //Note that when MPICH_ASYNC_PROGRESS is defined the thread-safety
    //level will automatically be set to MPI_THREAD_MULTIPLE.
    if((e = MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &provided)) != MPI_SUCCESS)
        EXCEPT_MPI(e);

    if((e = MPI_Comm_rank(MPI_COMM_WORLD, &pgrid_myrank)) != MPI_SUCCESS)
        EXCEPT_MPI(e);

    if((e = MPI_Comm_size(MPI_COMM_WORLD, &pgrid_worldsize)) != MPI_SUCCESS)
        EXCEPT_MPI(e);

    //Lets do CPU bindings and print process information
    char hostname[1024];
    int hostnamelen;
    MPI_Get_processor_name(hostname,&hostnamelen);
    int ncpus = sysconf(_SC_NPROCESSORS_ONLN);
    char *nnodes_env = getenv("BH_CLUSTER_NNODES");
    int nnodes = -1;

    char buf[1024];
    buf[0] = '\0';
    if(ncpus > 0 && nnodes_env != NULL)
    {
        nnodes = atoi(nnodes_env);
        assert(nnodes > 0);
        assert(pgrid_worldsize % nnodes == 0);

        int ppn = pgrid_worldsize / nnodes;
        int node_rank = pgrid_myrank % ppn;
        assert(ncpus % ppn == 0);
        int cpu_per_proc = ncpus / ppn;
        assert(cpu_per_proc > 0);

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for(int i=0; i<cpu_per_proc; ++i)
            CPU_SET(i+node_rank*cpu_per_proc, &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

        int start = node_rank*cpu_per_proc;
        int end = start+cpu_per_proc-1;
        if(start == end)
            sprintf(buf+strlen(buf), " (cpu bindings: %d)", start);
        else
            sprintf(buf+strlen(buf), " (cpu bindings: %d-%d)", start, end);
    }

    printf("[CLUSTER] rank %d running on %s:%d%s\n",
            pgrid_myrank, hostname, sched_getcpu(), buf);

}/* pgrid_init */

/*===================================================================
 *
 * Finalize the MPI process grid.
 */
void pgrid_finalize(void)
{
    MPI_Finalize();
} /* pgrid_finalize */
