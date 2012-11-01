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

#include <mpi.h>
#include <cassert>
#include <cphvb.h>
#include "dispatch.h"

/* Send payload to all slave processes.
 * @type is the type of the message
 * @size is the size of the payload
 * @payload is the payload of the message
*/
cphvb_error dispatch_send(int type, int size, const void *payload)
{
    static int buf_size = 0;
    static char *buf = NULL;
    const int csize = CPHVB_CLUSTER_DISPATCH_CHUNKSIZE;
    const int msgsize = size + 2 * sizeof(int);
    int e;
    assert(csize > 2 * sizeof(int));

    //Expand the read buffer if need  
    if(buf_size < msgsize)
    {
        buf_size = msgsize;
        buf = (char*) realloc(buf, buf_size);
        if(buf == NULL)
           return CPHVB_OUT_OF_MEMORY;
    }

    //Create message
    dispatch_msg *msg = (dispatch_msg *)buf;
    msg->type = type;
    msg->size = size;
    memcpy(msg->payload, payload, size);
    
    if((e = MPI_Bcast(buf, csize, MPI_BYTE, 0, MPI_COMM_WORLD)) != 0)
        return CPHVB_ERROR;
       
    int size_left = msgsize - csize;
    if(size_left > 0)
    {
        if((e = MPI_Bcast(buf + csize, size_left, MPI_BYTE, 0, MPI_COMM_WORLD)) != 0)
            return CPHVB_ERROR;
    }
    return CPHVB_SUCCESS;
}

/* Receive payload from master process.
 * @msg the received message (should not be freed)
*/
cphvb_error dispatch_recv(dispatch_msg **msg)
{

    int e;
    const int csize = CPHVB_CLUSTER_DISPATCH_CHUNKSIZE;
    static int buf_size = csize;
    static char *buf = (char*) malloc(csize);
    if(buf == NULL)
        return CPHVB_OUT_OF_MEMORY;
    
    //Get header of the message
    if((e = MPI_Bcast(buf, csize, MPI_BYTE, 0, MPI_COMM_WORLD)) != 0)
        return CPHVB_ERROR;

    //Read total message size
    int size = ((dispatch_msg *)buf)->size;
 
    //Expand the read buffer if need  
    if(buf_size < size)
    {
        buf_size = size;
        realloc(buf, buf_size);
    }
 
    //Get rest of the message
    int size_left = size - csize;
    if(size_left > 0)
    {
        if((e = MPI_Bcast(buf + csize, size_left, MPI_BYTE, 0, MPI_COMM_WORLD)) != 0)
            return CPHVB_ERROR;

    }

    *msg = (dispatch_msg*) buf;
    
    return CPHVB_SUCCESS;
}

