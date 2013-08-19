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
#include <cassert>
#include <map>
#include "pgrid.h"
#include "array.h"
#include "exec.h"
#include "except.h"
#include "batch.h"
#include "tmp.h"


/* Gather or scatter the global base array between processes.
 * NB: this is a collective operation.
 *
 * @scatter If true we scatter else we gather
 * @global_ary Global base array
 */
static void comm_gather_scatter(int scatter, bh_base *global_ary)
{
    bh_error err;
    bh_base *local_ary = array_get_local(global_ary);
    bh_intp totalsize = global_ary->nelem;

    if(totalsize <= 0)
        return;

    //Find the local size for all processes
    int sendcnts[pgrid_worldsize], displs[pgrid_worldsize];
    {
        bh_intp s = totalsize / pgrid_worldsize;//local size for all but the last process
        s *= bh_type_size(global_ary->type);
        for(int i=0; i<pgrid_worldsize; ++i)
        {
            sendcnts[i] = s;
            displs[i] = s * i;
        }
        //The last process gets the rest
        sendcnts[pgrid_worldsize-1] += totalsize % pgrid_worldsize * bh_type_size(global_ary->type);
    }

    int e;
    if(scatter)
    {
        //The slave-processes may need to allocate memory
        if(sendcnts[pgrid_myrank] > 0 && local_ary->data == NULL)
        {
            if((err = bh_data_malloc(local_ary)) != BH_SUCCESS)
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
            if((err = bh_data_malloc(global_ary)) != BH_SUCCESS)
                EXCEPT_OUT_OF_MEMORY();
        }

        //We will always allocate the local array when gathering because
        //only the last process knows if the array has been initiated.
        if((err = bh_data_malloc(local_ary)) != BH_SUCCESS)
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


/* Distribute the global base array data to all slave processes.
 * The master-process MUST have allocated the @global_ary data.
 * NB: this is a collective operation.
 *
 * @global_ary Global base array
 */
void comm_master2slaves(bh_base *global_ary)
{
    batch_flush();
    comm_gather_scatter(1, global_ary);
}


/* Gather the global base array data at the master processes.
 * NB: this is a collective operation.
 *
 * @global_ary Global base array
 */
void comm_slaves2master(bh_base *global_ary)
{
    batch_flush();
    comm_gather_scatter(0, global_ary);
}


/* Communicate array data such that the processes can apply local computation.
 * NB: The process that owns the data and the process where the data is located
 *     must both call this function.
 *
 * @chunk          The local array chunk to communicate
 * @sending_rank   The rank of the sending process
 * @receiving_rank The rank of the receiving process, e.g. the process that should
 *                 apply the computation
 */
void comm_array_data(const bh_view &chunk, int sending_rank, int receiving_rank)
{
//    printf("comm_array_data %d => %d: ", sending_rank, receiving_rank);
//    bh_view v = chunk;
//    bh_pprint_array(&v);

    //Check if communication is even necessary
    if(sending_rank == receiving_rank)
        return;

    if(pgrid_myrank == receiving_rank)
    {
        //Schedule the receive message
        batch_schedule_comm(0, sending_rank, chunk);
    }
    else if(pgrid_myrank == sending_rank)
    {
        //We need to copy the local array view into a base array buffer.
        bh_view tmp_view = chunk;
        tmp_view.base = tmp_get_ary(bh_base_array(&chunk)->type,
                                    bh_nelements_nbcast(&chunk));
        tmp_view.start = 0;

        //Set a contiguous row-major stride while preserving the zero-strided
        //dimensions (i.e. preserving broadcasted dimensions).
        bh_intp s = 1;
        for(bh_intp i=chunk.ndim-1; i >= 0; --i)
        {
            if(tmp_view.stride[i] > 0)
            {
                tmp_view.stride[i] = s;
                s *= tmp_view.shape[i];
            }
        }

        //Tell the VEM to do the data copy.
        bh_view ops[] = {tmp_view, chunk};
        batch_schedule_inst(BH_IDENTITY, ops, NULL);

        //Schedule the send message
        batch_schedule_comm(1, receiving_rank, tmp_view);

        //Cleanup the local arrays
        batch_schedule_inst(BH_FREE, bh_base_array(&tmp_view));
        batch_schedule_inst(BH_DISCARD, bh_base_array(&tmp_view));
    }
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
void comm_array_data(const ary_chunk &chunk, int receiving_rank)
{
    comm_array_data(chunk.ary, chunk.rank, receiving_rank);
}




