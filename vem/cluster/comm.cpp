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

/* Distribute the global array data to all slave processes.
 * NB: this is a collective operation.
 * 
 * @global_ary Global base array
 */
cphvb_error comm_master2slaves(cphvb_array *global_ary)
{
    assert(global_ary->base == NULL);
    cphvb_intp totalsize = cphvb_array_size(global_ary);

    if(totalsize <= 0)
        return CPHVB_SUCCESS;

    int sendcnts[pgrid_worldsize], displs[pgrid_worldsize];
    cphvb_intp localsize = totalsize / pgrid_worldsize;//local size for all but the last process
    for(int i=0; i<pgrid_worldsize; ++i)
    {
        sendcnts[i] = localsize;
        displs[i] = localsize * i;
    }
    //The last process gets the rest
    sendcnts[pgrid_worldsize-1] += totalsize % pgrid_worldsize;

    if(pgrid_myrank == 0)
    {
        if(global_ary->data == NULL)    
        {
            fprintf(stderr, "Warning - comm_master2slaves() is communicating "
                            "an uninitiated global array\n");
            cphvb_data_malloc(global_ary);
        }
    }
    else if(global_ary->data == NULL)
    {
        global_ary->data = cphvb_memory_malloc(sendcnts[pgrid_myrank-1]);
    }

    //The root need not send data
    sendcnts[0] = 0;
    
    int e = MPI_Scatterv(global_ary->data, sendcnts, displs, MPI_BYTE, 
                         global_ary->data, sendcnts[pgrid_worldsize-1], MPI_BYTE, 
                         0, MPI_COMM_WORLD);
    if(e != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Scatterv() in comm_master2slaves() returns error: %d\n", e);
        return CPHVB_ERROR;
    }
  
    return CPHVB_SUCCESS;
}
