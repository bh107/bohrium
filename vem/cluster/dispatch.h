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
#include <stack>

/* Dispatch message type */
enum /* int */
{
    CPHVB_CLUSTER_DISPATCH_INIT,
    CPHVB_CLUSTER_DISPATCH_SHUTDOWN,
    CPHVB_CLUSTER_DISPATCH_EXEC,
    CPHVB_CLUSTER_DISPATCH_UFUNC
};


//The size of a message chunk in bytes
#define CPHVB_CLUSTER_DISPATCH_DEFAULT_MSG_SIZE (256)


//The header of a dispatch message. If 'size' is larger than the message 
//default size, consecutive payload messages will follow the dispatch message.
typedef struct
{
    //Message type
    int type;
    //Size of the payload in bytes
    int size;
    //The content of the message
    char payload[];
}dispatch_msg;

//ID extension to cphvb_array
typedef struct
{
    //The id of the array. This is identical with the array-struct address 
    //on the master-process.
    cphvb_intp id;
    //The array-struct.
    cphvb_array ary;
}dispatch_array;


/* Initiate the dispatch system. */
cphvb_error dispatch_reset(void);

    
/* Finalize the dispatch system. */
cphvb_error dispatch_finalize(void);

/* Insert the new array into the array store and the array maps.
 * 
 * @master_ary The master array to register locally
 * @return Pointer to the registered array.
 */
cphvb_array* dispatch_new_slave_array(const cphvb_array *master_ary, cphvb_intp master_id);


/* Get the slave array.
 *
 * @master_array_id The master array id, which is the data pointer 
 *                  in the address space of the master-process.
 * @return Pointer to the registered array.
 */
cphvb_array* dispatch_master2slave(cphvb_intp master_array_id);


/* Check if the slave array exist.
 *
 * @master_array_id The master array id, which is the data pointer 
 *                  in the address space of the master-process.
 * @return True when the slave array exist locally.
 */
bool dispatch_slave_exist(cphvb_intp master_array_id);


/* Register the array as known by all the slaves.
 * Note that this function is only used by the master
 *
 * @ary The array that now is known.
 */
void dispatch_slave_known_insert(cphvb_array *ary);


/* Check if the array is known by all the slaves.
 * Note that this function is only used by the master
 *
 * @ary The array that should be checked.
 * @return True if the array is known by all slave-processes
 */
bool dispatch_slave_known_check(cphvb_array *ary);


/* Remove the array as known by all the slaves.
 * Note that this function is only used by the master
 *
 * @ary The array that now is unknown.
 */
void dispatch_slave_known_remove(cphvb_array *ary);


/* Reserve memory on the send message payload.
 * @size is the number of bytes to reserve
 * @payload is the output pointer to the reserved memory
 */
cphvb_error dispatch_reserve_payload(cphvb_intp size, void **payload);


/* Add data to the send message payload.
 * @size is the size of the data in bytes
 * @data is the data to add to the send buffer
 */
cphvb_error dispatch_add2payload(cphvb_intp size, const void *data);


/* Receive payload from master process.
 * @msg the received message (should not be freed)
 */
cphvb_error dispatch_recv(dispatch_msg **msg);


/* Send payload to all slave processes.
 * @type is the type of the message
*/
cphvb_error dispatch_send(int type);


/* Broadcast array-data to all slaves.
 * NB: this is a collective operation.
 *
 * @arys the base-arrays in question.
*/
cphvb_error dispatch_array_data(std::stack<cphvb_array*> arys);


/* Dispatch an instruction list to the slaves, which include new array-structs.
 * @count is the number of instructions in the list
 * @inst_list is the instruction list
 */
cphvb_error dispatch_inst_list(cphvb_intp count,
                               const cphvb_instruction inst_list[]);

#endif
