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
#include <cassert>
#include "pgrid.h"
#include "array.h"


/* Gather or scatter the global array processes.
 * NB: this is a collective operation.
 * 
 * @scatter If true we scatter else we gather
 * @global_ary Global base array
 */
static cphvb_error gather_scatter(int scatter, cphvb_array *global_ary)
{
    assert(global_ary->base == NULL);
    cphvb_error err;
    cphvb_intp totalsize = cphvb_nelements(global_ary->ndim, global_ary->shape);

    if(totalsize <= 0)
        return CPHVB_SUCCESS;

    int sendcnts[pgrid_worldsize], displs[pgrid_worldsize];
    cphvb_intp localsize = totalsize / pgrid_worldsize;//local size for all but the last process
    localsize *= cphvb_type_size(global_ary->type);
    for(int i=0; i<pgrid_worldsize; ++i)
    {
        sendcnts[i] = localsize;
        displs[i] = localsize * i;
    }
    //The last process gets the rest
    sendcnts[pgrid_worldsize-1] += totalsize % pgrid_worldsize * cphvb_type_size(global_ary->type);
    
    //Get local array
    cphvb_array *local_ary = array_get_local(global_ary);

    //We may need to do some memory allocations
    if(pgrid_myrank == 0)
    {
        if(global_ary->data == NULL)    
        {
            if((err = cphvb_data_malloc(global_ary)) != CPHVB_SUCCESS)
                return err;
        }
    }
    if((err = cphvb_data_malloc(local_ary)) != CPHVB_SUCCESS)
         return err;
    
    int e;
    if(scatter)
    {
        e = MPI_Scatterv(global_ary->data, sendcnts, displs, MPI_BYTE, 
                         local_ary->data, sendcnts[pgrid_myrank], MPI_BYTE, 
                         0, MPI_COMM_WORLD);
    }
    else
    {
        e = MPI_Gatherv(local_ary->data, sendcnts[pgrid_myrank], MPI_BYTE, 
                        global_ary->data, sendcnts, displs, MPI_BYTE, 
                        0, MPI_COMM_WORLD);
    }
    if(e != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI error in gather_scatter() returns error: %d\n", e);
        return CPHVB_ERROR;
    }
    return CPHVB_SUCCESS;
}


/* Distribute the global array data to all slave processes.
 * NB: this is a collective operation.
 * 
 * @global_ary Global base array
 */
cphvb_error comm_master2slaves(cphvb_array *global_ary)
{
    return gather_scatter(1, global_ary);
}


/* Gather the global array data at the master processes.
 * NB: this is a collective operation.
 * 
 * @global_ary Global base array
 */
cphvb_error comm_slaves2master(cphvb_array *global_ary)
{
    return gather_scatter(0, global_ary);
}


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
                            int receiving_rank)
{
    //Check if communication is even necessary
    if(local_ary_ext->rank == receiving_rank)
        return CPHVB_SUCCESS;

    if(pgrid_myrank == receiving_rank)
    {
        //This array is temporary and
        //located contiguous in memory (row-major)
        local_ary->base = NULL;
        local_ary->start = 0;
        cphvb_intp s = 1;
        for(cphvb_intp i=local_ary->ndim-1; i >= 0; --i)
        {    
            local_ary->stride[i] = s;
            s *= local_ary->shape[i];
        }
        cphvb_intp size = s * cphvb_type_size(local_ary->type);
        local_ary->data = malloc(size);
        if(local_ary->data == NULL)
            return CPHVB_OUT_OF_MEMORY;
        MPI_Recv(local_ary->data, size, MPI_BYTE, local_ary_ext->rank, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if(pgrid_myrank == local_ary_ext->rank)
    {
        //Create the MPI type for sending
        int typesize = cphvb_type_size(local_ary->type);
        int d = local_ary->ndim-1;
        MPI_Datatype mpi_type, tmp;
        MPI_Type_vector(local_ary->shape[d], 
                        typesize, 
                        local_ary->stride[d] * typesize,
                        MPI_BYTE,
                        &mpi_type);
        while(--d >= 0)
        {
            MPI_Type_vector(local_ary->shape[d], 1, 
                            local_ary->stride[d],
                            mpi_type,
                            &tmp);
            MPI_Type_free(&mpi_type);
            mpi_type = tmp;
        }
        MPI_Type_commit(&mpi_type);
        char *base_data = (char*) cphvb_base_array(local_ary)->data;
        MPI_Send(base_data + local_ary->start * typesize,
                 1, mpi_type, receiving_rank, 0, MPI_COMM_WORLD);

        MPI_Type_free(&mpi_type);
    }
    return CPHVB_SUCCESS;
}




