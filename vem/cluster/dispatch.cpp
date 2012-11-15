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


//Message buffer, current offset and buffer size
static int buf_size=sizeof(dispatch_msg);
static dispatch_msg *msg=NULL;

/* Initiate the dispatch system. */
cphvb_error dispatch_reset(void)
{
    const int dms = CPHVB_CLUSTER_DISPATCH_DEFAULT_MSG_SIZE;
    assert(((int)dms) > sizeof(dispatch_msg));
    if(msg == NULL)
    {
        msg = (dispatch_msg*) malloc(dms);
        if(msg == NULL)
            return CPHVB_OUT_OF_MEMORY;
        buf_size = dms;
    }
    msg->size = 0;
    return CPHVB_SUCCESS;
}

    
/* Finalize the dispatch system. */
cphvb_error dispatch_finalize(void)
{
    free(msg);
    return CPHVB_SUCCESS;
}


/* Add data to the send message payload.
 * @size is the size of the data in bytes
 * @data is the data to add to the send buffer
 */
cphvb_error dispatch_add2payload(cphvb_intp size, void *data)
{
    cphvb_intp new_msg_size = sizeof(dispatch_msg) + msg->size + size;
    //Expand the buffer if need
    if(buf_size < new_msg_size)
    {
        buf_size = new_msg_size*2;
        msg = (dispatch_msg*) realloc(msg, buf_size);
        if(msg == NULL)
           return CPHVB_OUT_OF_MEMORY;
    }
    memcpy(msg->payload+msg->size, data, size);
    msg->size += size;
    return CPHVB_SUCCESS;
}


/* Send payload to all slave processes.
 * @type is the type of the message
*/
cphvb_error dispatch_send(int type)
{
    const int dms = CPHVB_CLUSTER_DISPATCH_DEFAULT_MSG_SIZE;
    int e;

    //Create message
    msg->type = type;
        
    if((e = MPI_Bcast(msg, dms, MPI_BYTE, 0, MPI_COMM_WORLD)) != 0)
        return CPHVB_ERROR;
       
    int size_left = sizeof(dispatch_msg) + msg->size - dms;
    if(size_left > 0)
    {
        if((e = MPI_Bcast(msg + dms, size_left, MPI_BYTE, 0, MPI_COMM_WORLD)) != 0)
            return CPHVB_ERROR;
    }
    return CPHVB_SUCCESS;
}


/* Receive payload from master process.
 * @msg the received message (should not be freed)
*/
cphvb_error dispatch_recv(dispatch_msg **message)
{

    int e;
    const int dms = CPHVB_CLUSTER_DISPATCH_DEFAULT_MSG_SIZE;
    
    //Get header of the message
    if((e = MPI_Bcast(msg, dms, MPI_BYTE, 0, MPI_COMM_WORLD)) != 0)
        return CPHVB_ERROR;

    //Read total message size
    int size = sizeof(dispatch_msg) + msg->size;
 
    //Expand the read buffer if need  
    if(buf_size < size)
    {
        buf_size = size;
        msg = (dispatch_msg*) realloc(msg, buf_size);
        if(msg == NULL)
           return CPHVB_OUT_OF_MEMORY;
    }
 
    //Get rest of the message
    int size_left = size - dms;
    if(size_left > 0)
    {
        if((e = MPI_Bcast(msg + dms, size_left, MPI_BYTE, 0, MPI_COMM_WORLD)) != 0)
            return CPHVB_ERROR;
    }

    *message = msg;
    return CPHVB_SUCCESS;
}

