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

#include <mpi.h>
#include <cassert>
#include <map>
#include <set>
#include <bh.h>
#include <StaticStore.hpp>
#include "dispatch.h"
#include "pgrid.h"
#include "except.h"
#include "comm.h"
#include "timing.h"


//Message buffer, current offset and buffer size
static int buf_size=sizeof(dispatch_msg);
static dispatch_msg *msg=NULL;

//Maps for translating between arrays addressing the master process
//and arrays addressing a slave process.
static std::map<bh_intp, bh_base*> map_master2slave;
static std::map<bh_base*,bh_intp> map_slave2master;
static StaticStore<bh_base> slave_ary_store(512);
static std::set<bh_base*> slave_known_arrays;


/* Initiate the dispatch system. */
void dispatch_reset(void)
{
    const int dms = BH_CLUSTER_DISPATCH_DEFAULT_MSG_SIZE;
    assert(((int)dms) > sizeof(dispatch_msg));
    if(msg == NULL)
    {
        msg = (dispatch_msg*) malloc(dms);
        if(msg == NULL)
            EXCEPT_OUT_OF_MEMORY();
        buf_size = dms;
    }
    msg->size = 0;
}


/* Finalize the dispatch system. */
void dispatch_finalize(void)
{
    free(msg);
}


/* Insert the new array into the array store and the array maps.
 * Note that this function is only used by the slaves
 *
 * @master_ary  The master array to register locally
 * @return      Pointer to the registered array.
 */
bh_base* dispatch_new_slave_array(const bh_base *master_ary, bh_intp master_id)
{
    assert(pgrid_myrank > 0);
    bh_base *ary = slave_ary_store.c_next();
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
 * @return          Pointer to the registered array.
 */
bh_base* dispatch_master2slave(bh_intp master_array_id)
{
    assert(pgrid_myrank > 0);
    return map_master2slave[master_array_id];
}


/* Check if the slave array exist.
 * Note that this function is only used by the slaves
 *
 * @master_array_id The master array id, which is the data pointer
 *                  in the address space of the master-process.
 * @return          True when the slave array exist locally.
 */
bool dispatch_slave_exist(bh_intp master_array_id)
{
    assert(pgrid_myrank > 0);
    return map_master2slave.count(master_array_id) > 0;
}


/* Register the array as known by all the slaves.
 * Note that this function is only used by the master
 *
 * @ary The array that now is known.
 */
void dispatch_slave_known_insert(bh_base *ary)
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
bool dispatch_slave_known_check(bh_base *ary)
{
    assert(pgrid_myrank == 0);
    return slave_known_arrays.count(ary) > 0;
}


/* Remove the array as known by all the slaves.
 * Note that this function is used both by the master and slaves
 *
 * @ary The array that now is unknown.
 */
void dispatch_slave_known_remove(bh_base *ary)
{
    if(pgrid_myrank == 0)
    {
        assert(dispatch_slave_known_check(ary));
        slave_known_arrays.erase(ary);
    }
    else
    {
        assert(map_slave2master.count(ary) == 1);
        bh_intp master_id = map_slave2master[ary];
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
void dispatch_reserve_payload(bh_intp size, void **payload)
{
    bh_intp new_msg_size = sizeof(dispatch_msg) + msg->size + size;
    //Expand the buffer if need
    if(buf_size < new_msg_size)
    {
        buf_size = new_msg_size*2;
        msg = (dispatch_msg*) realloc(msg, buf_size);
        if(msg == NULL)
            EXCEPT_OUT_OF_MEMORY();
    }
    *payload = msg->payload + msg->size;
    msg->size += size;
}


/* Add data to the send message payload.
 * @size is the size of the data in bytes
 * @data is the data to add to the send buffer
 */
void dispatch_add2payload(bh_intp size, const void *data)
{
    void *payload;
    //Reserve memory on the send message
    dispatch_reserve_payload(size, &payload);

    //Copy 'data' to the send message
    memcpy(payload, data, size);
}


/* Send payload to all slave processes.
 * @type is the type of the message
*/
void dispatch_send(int type)
{
    const int dms = BH_CLUSTER_DISPATCH_DEFAULT_MSG_SIZE;
    int e;

    //Create message
    msg->type = type;

    if((e = MPI_Bcast(msg, dms, MPI_BYTE, 0, MPI_COMM_WORLD)) != MPI_SUCCESS)
        EXCEPT_MPI(e);

    int size_left = sizeof(dispatch_msg) + msg->size - dms;
    if(size_left > 0)
    {
        if((e = MPI_Bcast(((char*)msg) + dms, size_left, MPI_BYTE, 0,
            MPI_COMM_WORLD)) != MPI_SUCCESS)
            EXCEPT_MPI(e);
    }
}


/* Receive payload from master process.
 * @msg the received message (should not be freed)
*/
void dispatch_recv(dispatch_msg **message)
{
    int e;
    const int dms = BH_CLUSTER_DISPATCH_DEFAULT_MSG_SIZE;

    bh_uint64 stime = bh_timing();

    //Get header of the message
    if((e = MPI_Bcast(msg, dms, MPI_BYTE, 0, MPI_COMM_WORLD)) != MPI_SUCCESS)
        EXCEPT_MPI(e);

    //Read total message size
    int size = sizeof(dispatch_msg) + msg->size;

    //Expand the read buffer if need
    if(buf_size < size)
    {
        buf_size = size;
        msg = (dispatch_msg*) realloc(msg, buf_size);
        if(msg == NULL)
           EXCEPT_OUT_OF_MEMORY();
    }

    //Get rest of the message
    int size_left = size - dms;
    if(size_left > 0)
    {
        if((e = MPI_Bcast(((char*)msg) + dms, size_left, MPI_BYTE, 0,
            MPI_COMM_WORLD)) != MPI_SUCCESS)
            EXCEPT_MPI(e);
    }
    *message = msg;
    bh_timing_save(timing_dispatch, stime, bh_timing());
}


/* Broadcast array-data to all slaves.
 * NB: this is a collective operation.
 *
 * @arys the base-arrays in question.
*/
void dispatch_array_data(std::stack<bh_base*> &arys)
{
    bh_uint64 stime = bh_timing();

    while (!arys.empty())
    {
        bh_base *ary = arys.top();
        if(ary->data != NULL)
        {
            if(pgrid_myrank != 0)
                ary->data = NULL;//The data-pointer was referencing memory on the master-process.
            comm_master2slaves(ary);
        }
        arys.pop();
    }
    bh_timing_save(timing_dispatch_array_data, stime, bh_timing());
}


/* Dispatch the BhIR to the slaves, which includes new array-structs.
 * @bhir  The BhIR in question
 */
void dispatch_bhir(const bh_ir *bhir)
{
    dispatch_reset();

    bh_uint64 stime = bh_timing();

    /* The execution message has the form:
     * 1   x bh_ir bhir  //serialized BhIR
     * 1   x bh_intp NOA //number of new arrays
     * NOA x dispatch_array //list of new arrays unknown to the slaves
     */

    //Pack the BhIR.
    void *msg_bhir;
    dispatch_reserve_payload(bh_ir_totalsize(bhir), &msg_bhir);
    bh_ir_serialize(msg_bhir, bhir);

    //Make reservation for the number of new arrays (NOA).
    bh_intp msg_noa_offset, noa=0;
    void *msg_noa;
    dispatch_reserve_payload(sizeof(bh_intp), &msg_noa);

    //We need a message offset instead of a pointer since dispatch_reserve_payload() may
    //re-allocate the 'msg_noa' pointer at a later time.
    msg_noa_offset = msg->size - sizeof(bh_intp);

    //Stack of new base arrays
    std::stack<bh_base*> base_darys;

    //Pack the array list.
    for(bh_intp i=0; i<bhir->ninstr; ++i)
    {
        const bh_instruction *inst = &bhir->instr_list[i];
        int nop = bh_operands_in_instruction(inst);
        for(bh_intp j=0; j<nop; ++j)
        {
            bh_view *operand = &bh_inst_operands((bh_instruction*)inst)[j];
            if(bh_is_constant(operand))
                continue;//No need to dispatch constants

            bh_base *base = bh_base_array(operand);
            if(!dispatch_slave_known_check(base))//The array is unknown to the slaves.
            {
                dispatch_array *dary;
                dispatch_reserve_payload(sizeof(dispatch_array),(void**) &dary);
                dispatch_slave_known_insert(base);
                //The master-process's memory pointer is the id of the array.
                dary->id = (bh_intp) base;
                dary->ary = *base;
                ++noa;
                base_darys.push(base);
            }
        }
    }
    //Save the number of new arrays
    *((bh_intp*)(msg->payload+msg_noa_offset)) = noa;

    //Dispath the execution message
    dispatch_send(BH_CLUSTER_DISPATCH_EXEC);

    //Dispath the array data
    dispatch_array_data(base_darys);

    bh_timing_save(timing_dispatch, stime, bh_timing());
}
