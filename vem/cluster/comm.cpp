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
#include <map>
#include "pgrid.h"
#include "array.h"
#include "exec.h"
#include "except.h"
#include "batch.h"


/* Gather or scatter the global array processes.
 * NB: this is a collective operation.
 * 
 * @scatter If true we scatter else we gather
 * @global_ary Global base array
 */
static void gather_scatter(int scatter, cphvb_array *global_ary)
{
    assert(global_ary->base == NULL);
    cphvb_error err;
    cphvb_array *local_ary = array_get_local(global_ary);
    cphvb_intp totalsize = cphvb_nelements(global_ary->ndim, global_ary->shape);

    if(totalsize <= 0)
        return;

    //Find the local size for all processes
    int sendcnts[pgrid_worldsize], displs[pgrid_worldsize];
    {
        cphvb_intp s = totalsize / pgrid_worldsize;//local size for all but the last process
        s *= cphvb_type_size(global_ary->type);
        for(int i=0; i<pgrid_worldsize; ++i)
        {
            sendcnts[i] = s;
            displs[i] = s * i;
        }
        //The last process gets the rest
        sendcnts[pgrid_worldsize-1] += totalsize % pgrid_worldsize * cphvb_type_size(global_ary->type);
    }

    int e;
    if(scatter)
    {
        //The slave-processes may need to allocate memory
        if(sendcnts[pgrid_myrank] > 0 && local_ary->data == NULL)
        {
            if((err = cphvb_data_malloc(local_ary)) != CPHVB_SUCCESS)
                EXCEPT_OUT_OF_MEMORY();
        }
        //The master-process MUST have allocated memory already
        assert(pgrid_myrank != 0 || global_ary->data != NULL);
        
        //Scatter from master to slaves
        e = MPI_Scatterv(global_ary->data, sendcnts, displs, MPI_BYTE, 
                         local_ary->data, sendcnts[pgrid_myrank], MPI_BYTE, 
                         0, MPI_COMM_WORLD);
    }
    else
    {
        //The master-processes may need to allocate memory
        if(pgrid_myrank == 0 && global_ary->data == NULL)
        {
            if((err = cphvb_data_malloc(global_ary)) != CPHVB_SUCCESS)
                EXCEPT_OUT_OF_MEMORY();
        }
        
        //We will always allocate the local array when gathering because 
        //only the last process knows if the array has been initiated.
        if((err = cphvb_data_malloc(local_ary)) != CPHVB_SUCCESS)
            EXCEPT_OUT_OF_MEMORY();
    
        assert(sendcnts[pgrid_myrank] == 0 || local_ary->data != NULL);
        
        //Gather from the slaves to the master
        e = MPI_Gatherv(local_ary->data, sendcnts[pgrid_myrank], MPI_BYTE, 
                        global_ary->data, sendcnts, displs, MPI_BYTE, 
                        0, MPI_COMM_WORLD);
    }
    if(e != MPI_SUCCESS)
        EXCEPT_MPI(e);        
}


/* Distribute the global array data to all slave processes.
 * The master-process MUST have allocated the @global_ary data.
 * NB: this is a collective operation.
 * 
 * @global_ary Global base array
 */
void comm_master2slaves(cphvb_array *global_ary)
{
    gather_scatter(1, global_ary);
}


/* Gather the global array data at the master processes.
 * NB: this is a collective operation.
 * 
 * @global_ary Global base array
 */
void comm_slaves2master(cphvb_array *global_ary)
{
    gather_scatter(0, global_ary);
}


/* Communicate array data such that the processes can apply local computation.
 * This function may reshape the input array chunk.
 * NB: The process that owns the data and the process where the data is located
 *     must both call this function.
 *     
 * @chunk The local array chunk to communicate
 * @receiving_rank The rank of the receiving process, e.g. the process that should
 *                 apply the computation
 */
void comm_array_data(ary_chunk *chunk, int receiving_rank)
{
    cphvb_error e;
    cphvb_array *local_ary = chunk->ary;
    int rank = chunk->rank;

    //Check if communication is even necessary
    if(rank == receiving_rank)
        return;

    if(pgrid_myrank == receiving_rank)
    {
        //This array is temporary and
        //located contiguous in memory (row-major)
        assert(local_ary->base == NULL);
        assert(local_ary->start == 0);
        assert(local_ary->data == NULL);
        if((e = cphvb_data_malloc(local_ary)) != CPHVB_SUCCESS)
            EXCEPT_OUT_OF_MEMORY();
        
        MPI_Recv(local_ary->data, cphvb_array_size(local_ary), MPI_BYTE, 
                 rank, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if(pgrid_myrank == rank)
    {
        //We need to copy the local array view into a base array.
        cphvb_array *tmp_ary = batch_tmp_ary();
        *tmp_ary = *local_ary;
        tmp_ary->base  = NULL;
        tmp_ary->data  = NULL;
        tmp_ary->start = 0;
   
        //Compute a row-major stride for the tmp array.
        cphvb_intp nelem = 1;
        for(cphvb_intp i=tmp_ary->ndim-1; i >= 0; --i)
        {    
            tmp_ary->stride[i] = nelem;
            nelem *= tmp_ary->shape[i];
        }

        //Tell the VEM to do the data copy.
        cphvb_array *ops[] = {tmp_ary, local_ary};
        exec_local_inst(CPHVB_IDENTITY, ops, NULL);
        assert(tmp_ary->data != NULL);
        MPI_Send(tmp_ary->data, nelem * cphvb_type_size(tmp_ary->type), 
                 MPI_BYTE, receiving_rank, 0, MPI_COMM_WORLD);

        //Cleanup the local arrays
        exec_local_inst(CPHVB_FREE, &tmp_ary, NULL);
        exec_local_inst(CPHVB_DISCARD, &tmp_ary, NULL);
        exec_local_inst(CPHVB_DISCARD, &local_ary, NULL);
    }
}




