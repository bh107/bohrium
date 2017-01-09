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

#ifndef __BH_VEM_CLUSTER_DISPATCH_H
#define __BH_VEM_CLUSTER_DISPATCH_H

#include <stack>

/* Dispatch message type */
enum /* int */
{
    BH_CLUSTER_DISPATCH_INIT,
    BH_CLUSTER_DISPATCH_SHUTDOWN,
    BH_CLUSTER_DISPATCH_EXEC,
    BH_CLUSTER_DISPATCH_EXTMETHOD
};


//The size of a message chunk in bytes
#define BH_CLUSTER_DISPATCH_DEFAULT_MSG_SIZE (256)


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

//ID extension to bh_array
typedef struct
{
    //The id of the array. This is identical with the base-struct address
    //on the master-process.
    bh_intp id;
    //The base-struct.
    bh_base ary;
}dispatch_array;


/* Initiate the dispatch system. */
void dispatch_reset(void);


/* Finalize the dispatch system. */
void dispatch_finalize(void);


/* Insert the new array into the array store and the array maps.
 * Note that this function is only used by the slaves
 *
 * @master_ary  The master array to register locally
 * @return      Pointer to the registered array.
 */
bh_base* dispatch_new_slave_array(const bh_base *master_ary, bh_intp master_id);


/* Get the slave array.
 * Note that this function is only used by the slaves
 *
 * @master_array_id The master array id, which is the data pointer
 *                  in the address space of the master-process.
 * @return          Pointer to the registered array.
 */
bh_base* dispatch_master2slave(bh_intp master_array_id);


/* Check if the slave array exist.
 * Note that this function is only used by the slaves
 *
 * @master_array_id The master array id, which is the data pointer
 *                  in the address space of the master-process.
 * @return          True when the slave array exist locally.
 */
bool dispatch_slave_exist(bh_intp master_array_id);


/* Register the array as known by all the slaves.
 * Note that this function is only used by the master
 *
 * @ary The array that now is known.
 */
void dispatch_slave_known_insert(bh_base *ary);


/* Check if the array is known by all the slaves.
 * Note that this function is only used by the master
 *
 * @ary The array that should be checked.
 * @return True if the array is known by all slave-processes
 */
bool dispatch_slave_known_check(bh_base *ary);


/* Remove the array as known by all the slaves.
 * Note that this function is used both by the master and slaves
 *
 * @ary The array that now is unknown.
 */
void dispatch_slave_known_remove(bh_base *ary);


/* Reserve memory on the send message payload.
 * @size is the number of bytes to reserve
 * @payload is the output pointer to the reserved memory
 */
void dispatch_reserve_payload(bh_intp size, void **payload);


/* Add data to the send message payload.
 * @size is the size of the data in bytes
 * @data is the data to add to the send buffer
 */
void dispatch_add2payload(bh_intp size, const void *data);


/* Send payload to all slave processes.
 * @type is the type of the message
*/
void dispatch_send(int type);


/* Receive payload from master process.
 * @msg the received message (should not be freed)
*/
void dispatch_recv(dispatch_msg **message);


/* Broadcast array-data to all slaves.
 * NB: this is a collective operation.
 *
 * @arys the base-arrays in question.
*/
void dispatch_array_data(std::stack<bh_base*> &arys);


/* Dispatch the BhIR to the slaves, which includes new array-structs.
 * @bhir  The BhIR in question
 */
void dispatch_bhir(bh_ir *bhir);

#endif
