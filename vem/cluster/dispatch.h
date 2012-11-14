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

#ifndef __CPHVB_VEM_CLUSTER_DISPATCH_H

#include <cphvb.h>

/* Dispatch message type */
enum /* int */
{
    CPHVB_CLUSTER_DISPATCH_INIT,
    CPHVB_CLUSTER_DISPATCH_SHUTDOWN,
    CPHVB_CLUSTER_DISPATCH_EXEC,
    CPHVB_CLUSTER_DISPATCH_UFUNC
};

//The size of a message chunk in bytes
#define CPHVB_CLUSTER_DISPATCH_DEFAULT_MSG_SIZE (1024)

//The header of a dispatch message. If 'size' is larger than the message 
//chunk size, consecutive payload messages will follow the dispatch message.
typedef struct
{
    //Message type
    int type;
    //Total message size (all included)
    int size;
    //The content of the message
    char payload[];
}dispatch_msg;


/* Send payload to all slave processes.
 * @type is the type of the message
 * @size is the size of the payload
 * @payload is the payload of the message
*/
cphvb_error dispatch_send(int type, int size, const void *payload);


/* Receive payload from master process.
 * @msg the received message (should not be freed)
*/
cphvb_error dispatch_recv(dispatch_msg **msg);

/* Initiate the dispatch system. */
cphvb_error dispatch_init(void);

    
/* Finalize the dispatch system. */
cphvb_error dispatch_finalize(void);

#endif
