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
#include <map>
#include <set>
#include <cphvb.h>
#include <StaticStore.hpp>
#include "dispatch.h"
#include "pgrid.h"
#include "comm.h"


//Message buffer, current offset and buffer size
static int buf_size=sizeof(dispatch_msg);
static dispatch_msg *msg=NULL;

//Maps for translating between arrays addressing the master process
//and arrays addressing a slave process.
static std::map<cphvb_intp, cphvb_array*> map_master2slave;
static std::map<cphvb_array*,cphvb_intp> map_slave2master;
static StaticStore<cphvb_array> slave_ary_store(512);
static std::set<cphvb_array*> slave_known_arrays;


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


/* Insert the new array into the array store and the array maps.
 * Note that this function is only used by the slaves
 * 
 * @master_ary The master array to register locally
 * @return Pointer to the registered array.
 */
cphvb_array* dispatch_new_slave_array(const cphvb_array *master_ary, cphvb_intp master_id)
{
    assert(pgrid_myrank > 0);
    cphvb_array *ary = slave_ary_store.c_next();
    *ary = *master_ary;
    
    assert(map_master2slave.count(master_id) == 0);
    assert(map_slave2master.count(ary) == 0);

    map_master2slave[master_id] = ary;
    map_slave2master[ary] = master_id;
    return ary;
}


/* Get the slave array.
 * Note that this function is only used by the slaves
 *
 * @master_array_id The master array id, which is the data pointer 
 *                  in the address space of the master-process.
 * @return Pointer to the registered array.
 */
cphvb_array* dispatch_master2slave(cphvb_intp master_array_id)
{
    assert(pgrid_myrank > 0);
    return map_master2slave[master_array_id];
}


/* Check if the slave array exist.
 * Note that this function is only used by the slaves
 *
 * @master_array_id The master array id, which is the data pointer 
 *                  in the address space of the master-process.
 * @return True when the slave array exist locally.
 */
bool dispatch_slave_exist(cphvb_intp master_array_id)
{
    assert(pgrid_myrank > 0);
    return map_master2slave.count(master_array_id) > 0;
}


/* Register the array as known by all the slaves.
 * Note that this function is only used by the master
 *
 * @ary The array that now is known.
 */
void dispatch_slave_known_insert(cphvb_array *ary)
{
    assert(pgrid_myrank == 0);
    slave_known_arrays.insert(ary);
}


/* Check if the array is known by all the slaves.
 * Note that this function is only used by the master
 *
 * @ary The array that should be checked.
 * @return True if the array is known by all slave-processes
 */
bool dispatch_slave_known_check(cphvb_array *ary)
{
    assert(pgrid_myrank == 0);
    return slave_known_arrays.count(ary) > 0;
}


/* Remove the array as known by all the slaves.
 * Note that this function is used both by the master and slaves
 *
 * @ary The array that now is unknown.
 */
void dispatch_slave_known_remove(cphvb_array *ary)
{
    if(pgrid_myrank == 0)
    {
        assert(dispatch_slave_known_check(ary));
        slave_known_arrays.erase(ary);
    }
    else
    {
        assert(map_slave2master.count(ary) == 1);
        cphvb_intp master_id = map_slave2master[ary];
        map_slave2master.erase(ary);
        assert(map_master2slave.count(master_id) == 1);
        map_master2slave.erase(master_id);
        slave_ary_store.erase(ary);
    }
}    


/* Reserve memory on the send message payload.
 * @size is the number of bytes to reserve
 * @payload is the output pointer to the reserved memory
 */
cphvb_error dispatch_reserve_payload(cphvb_intp size, void **payload)
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
    *payload = msg->payload + msg->size;
    msg->size += size;
    return CPHVB_SUCCESS;
}


/* Add data to the send message payload.
 * @size is the size of the data in bytes
 * @data is the data to add to the send buffer
 */
cphvb_error dispatch_add2payload(cphvb_intp size, const void *data)
{
    cphvb_error e;
    void *payload;
    //Reserve memory on the send message
    if((e  = dispatch_reserve_payload(size, &payload)) != CPHVB_SUCCESS)
        return e;
    //Copy 'data' to the send message
    memcpy(payload, data, size);
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
        if((e = MPI_Bcast(((char*)msg) + dms, size_left, MPI_BYTE, 0, MPI_COMM_WORLD)) != 0)
        {
            fprintf(stderr, "MPI_Bcast in dispatch_send() returns error: %d\n", e);
            return CPHVB_ERROR;
        }
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
        if((e = MPI_Bcast(((char*)msg) + dms, size_left, MPI_BYTE, 0, MPI_COMM_WORLD)) != 0)
        {
            fprintf(stderr, "MPI_Bcast in dispatch_recv() returns error: %d\n", e);
            return CPHVB_ERROR;
        }
    }

    *message = msg;
    return CPHVB_SUCCESS;
}


/* Broadcast array-data to all slaves.
 * NB: this is a collective operation.
 *
 * @arys the base-arrays in question.
*/
cphvb_error dispatch_array_data(std::stack<cphvb_array*> arys)
{
    cphvb_error e;
    while (!arys.empty())
    {
        cphvb_array *ary = arys.top();
        assert(ary->base == NULL);
        if(ary->data != NULL)
        {
            if(pgrid_myrank != 0)
                ary->data = NULL;//The data-pointer was referencing memory on the master-process.
            
            if((e = comm_master2slaves(ary)) != CPHVB_SUCCESS)
                return e;
        }
        arys.pop();
    } 
    return CPHVB_SUCCESS;
}


/* Dispatch an instruction list to the slaves, which includes new array-structs.
 * @count is the number of instructions in the list
 * @inst_list is the instruction list
 */
cphvb_error dispatch_inst_list(cphvb_intp count,
                               const cphvb_instruction inst_list[])
{
    cphvb_error e;

    if((e = dispatch_reset()) != CPHVB_SUCCESS)
        return e;

    /* The execution message has the form:
     * 1   x cphvb_intp NOI //number of instructions
     * NOI x cphvb_instruction //instruction list
     * 1   x cphvb_intp NOA //number of new arrays
     * NOA x dispatch_array //list of new arrays unknown to the slaves
     * 1   x cphvb_intp NOU //number of user-defined funstions
     * NOU x cphvb_userfunc //list of user-defined funstions
     */

    //Pack the number of instructions (NOI).
    if((e = dispatch_add2payload(sizeof(cphvb_intp), &count)) != CPHVB_SUCCESS)
        return e;
    
    //Pack the instruction list.
    if((e = dispatch_add2payload(count * sizeof(cphvb_instruction), inst_list)) != CPHVB_SUCCESS)
        return e;

    //Make reservation for the number of new arrays (NOA).
    cphvb_intp msg_noa_offset, noa=0;
    char *msg_noa;
    if((e = dispatch_reserve_payload(sizeof(cphvb_intp), (void**) &msg_noa)) != CPHVB_SUCCESS)
        return e;

    //We need a message offset instead of a pointer since dispatch_reserve_payload() may 
    //re-allocate the 'msg_noa' pointer at a later time.
    msg_noa_offset = msg->size - sizeof(cphvb_intp);

    //Stack of new base arrays
    std::stack<cphvb_array*> base_darys;

    //Pack the array list.
    for(cphvb_intp i=0; i<count; ++i)
    {
        const cphvb_instruction *inst = &inst_list[i];
        int nop = cphvb_operands_in_instruction(inst);
        for(cphvb_intp j=0; j<nop; ++j)
        {
            cphvb_array *op;
            if(inst->opcode == CPHVB_USERFUNC)
                op = inst->userfunc->operand[j];
            else
                op = inst->operand[j];
 
            if(cphvb_is_constant(op))
                continue;//No need to dispatch constants

            if(!dispatch_slave_known_check(op))//The array is unknown to the slaves.
            {   
                dispatch_array *dary;
                if((e = dispatch_reserve_payload(sizeof(dispatch_array),(void**) &dary)) 
                     != CPHVB_SUCCESS)
                    return e;
                dispatch_slave_known_insert(op);
                //The master-process's memory pointer is the id of the array.
                dary->id = (cphvb_intp) op;
                dary->ary = *op;
                ++noa;
                if(op->base == NULL)
                {
                    base_darys.push(op);
                }
                else if(!dispatch_slave_known_check(op->base))//Also check the base-array.
                {
                    if((e = dispatch_reserve_payload(sizeof(dispatch_array),(void**) &dary)) 
                         != CPHVB_SUCCESS)
                        return e;
                    dispatch_slave_known_insert(op->base);
                    dary->id = (cphvb_intp) op->base;
                    dary->ary = *op->base;
                    ++noa;
                    assert(dary->ary.base == NULL);
                    base_darys.push(op->base);
                }
            }
        }
    }
    //Save the number of new arrays
    *((cphvb_intp*)(msg->payload+msg_noa_offset)) = noa;

    //Make reservation for the number of new arrays (NOU).
    cphvb_intp msg_nou_offset, nou=0;
    char *msg_nou;
    if((e = dispatch_reserve_payload(sizeof(cphvb_intp), (void**) &msg_nou)) != CPHVB_SUCCESS)
        return e;

    //Again we need a message offset. 
    msg_nou_offset = msg->size - sizeof(cphvb_intp);

    //Pack the user-defined function list
    for(cphvb_intp i=0; i<count; ++i)
    {
        const cphvb_instruction *inst = &inst_list[i];
        if(inst->opcode == CPHVB_USERFUNC)
        {
            if((e = dispatch_add2payload(inst->userfunc->struct_size, 
                                         inst->userfunc)) != CPHVB_SUCCESS)
                return e;
            ++nou;
        }
    }
    //Save the number of user-defined functions 
    *((cphvb_intp*)(msg->payload+msg_nou_offset)) = nou;

    //Dispath the execution message
    if((e = dispatch_send(CPHVB_CLUSTER_DISPATCH_EXEC)) != CPHVB_SUCCESS)
        return e;

    //Dispath the array data
    if((e = dispatch_array_data(base_darys)) != CPHVB_SUCCESS)
        return e;

    return CPHVB_SUCCESS;
}
