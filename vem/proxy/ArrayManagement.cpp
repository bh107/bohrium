/*
 * ArrayManagement.cpp
 *
 *  Created on: Feb 6, 2014
 *      Author: d
 *
 * based on dispatch.h and dispatch.cpp
 *
 * Notes: Currently SYNC messages are received onto the actual data memory space
 * the original data is. This is done to save memory that would otherwise have to be used to store
 * the extra array. However, if network issues cause a receive error during data transfer, the original
 * matrix is also lost.
 */

#include <map>
#include <stack>
#include <set>

#include "ArrayManagement.h"
#include "BasicPacket.h"

#define DEFAULT_MSG_SIZE 256

// message buffer and size
static int buf_size = 0;
static bh_packet * msg = NULL;

// file descriptor of server
static int fd = 0;

static std::set<bh_base*> server_known_arrays;

void ArrayMan_init()
{
    msg = AllocatePacket(DEFAULT_MSG_SIZE);
    msg->header.packetID = BH_PTC_EXECUTE;
}

void ArrayMan_shutdown()
{
    FreePacket(msg);
}

void ArrayMan_server_known_insert(bh_base *ary)
{
    server_known_arrays.insert(ary);
}

bool ArrayMan_server_known_check(bh_base *ary)
{
    return server_known_arrays.count(ary) > 0;
}

void ArrayMan_server_known_remove(bh_base *ary)
{
    server_known_arrays.erase(ary);
}

bh_error sendArrayData(std::stack<bh_base*> arys, int filedes)
{
    bh_packet packet;
    packet.header.packetID = BH_PTC_ARRAY;
    while (!arys.empty())
    {
        bh_base *ary = arys.top();
        if(ary->data != NULL)
        {
            packet.header.packetSize = bh_base_size(ary);
            //printf("Base size: %ld\n", packet.header.packetSize);
            packet.data = ary->data;
            // the base address is the key
            packet.header.key = (long)ary;
            if(SendPacket(&packet, filedes)!=BH_SRVR_SUCCESS)
                return BH_ERROR;
        }
        arys.pop();
    }

    return BH_SUCCESS;
}

bh_error ArrayMan_reset_msg()
{
    if(msg == NULL)
    {
        return BH_ERROR;
    }
    if(msg->data == NULL)
    {
        msg->data = malloc( DEFAULT_MSG_SIZE);
        if(msg->data == NULL)
            return BH_SRVR_OUT_OF_MEMORY;
        buf_size = DEFAULT_MSG_SIZE;
    }
    msg->header.packetSize = 0;
    msg->header.packetID = BH_PTC_EXECUTE;
    return BH_SUCCESS;
}

bh_error ArrayMan_reserve_payload(bh_intp size, void ** pointer)
{
    // total size is the payload + what will be added to the payload
    bh_intp new_msg_size = msg->header.packetSize + size;

    if(buf_size < new_msg_size)
    {
        while(buf_size < new_msg_size)
            buf_size = 2*(buf_size+DEFAULT_MSG_SIZE);

        msg->data = realloc(msg->data, buf_size);
        if(msg == NULL)
            return BH_OUT_OF_MEMORY;
    }

    *pointer = (void*) ((bh_intp)msg->data + (bh_intp)msg->header.packetSize);
    msg->header.packetSize += size;
    return BH_SUCCESS;
}

void ArrayMan_add_to_payload(bh_intp size, void * data)
{
    void *ptr;
    ArrayMan_reserve_payload(size, &ptr);

    memcpy(ptr, data, size);
}

bh_error ArrayMan_send_payload(packet_protocol ptc, int filedes)
{
    msg->header.packetID = ptc;
    return SendPacket(msg, filedes);
}


static bh_error inspect(bh_instruction *instr)
{
    if(instr->opcode == BH_SYNC)
    {
        //printf("SYNC\n");
        //receive the incoming matrix and handle it
        bh_base *ary = bh_base_array(&instr->operand[0]);
        if(!ArrayMan_server_known_check(ary))
        {
            printf("Trying to sync unknown to server array\n");
            return BH_ERROR;
        }

        bh_packet pack;
        // allocate memory if it hasn't been allocated yet
        if(ary->data == NULL)
            bh_data_malloc(ary);
        pack.data = ary->data;

        //bh_pprint_instr(instr);
        // could receive and check in a separate memory space. It is done like this to save memory
        int res = ReceiveArrayPacket(fd, &pack, bh_base_size(ary), BH_PTC_SYNC, (long)ary);
        if( res != BH_SRVR_SUCCESS)
        {
            printf("Failed to receive array during sync, error code: %d\n", res);

            return BH_ERROR;
        }

        //printf("SYNC complete\n");
    }
    else if(instr->opcode == BH_FREE)
    {
        //printf("FREE\n");
        bh_base *ary = bh_base_array(&instr->operand[0]);
        bh_data_free(ary);
    }
    else if(instr->opcode == BH_DISCARD)
    {
        //printf("DISCARD\n");
        bh_base *ary = bh_base_array(&instr->operand[0]);
        ArrayMan_server_known_remove(ary);
    }
    return BH_SUCCESS;
}

// Created the bh_ir message that is sent from the client to the server
// Resets the buffer, so make sure you send everything on it before using this!
bh_error ArrayMan_client_bh_ir_package(bh_ir * bhir, int filedes)
{

    int res = ArrayMan_reset_msg();
    if(res != BH_SUCCESS)
        return res;

    //bh_pprint_bhir(bhir);


    // add bh_ir to payload
    void *ptr;
    res = ArrayMan_reserve_payload(bh_ir_totalsize(bhir), &ptr);
    if(res != BH_SUCCESS)
        return res;
    bh_ir_serialize(ptr, bhir);

    //Make reservation for the number of new arrays (NOA).
    bh_intp msg_noa_offset, noa=0;
    void *msg_noa;
    res = ArrayMan_reserve_payload(sizeof(bh_intp), &msg_noa);
    if(res != BH_SUCCESS)
            return res;

    //We need a message offset instead of a pointer since ArrayMan_reserve_payload() may
    //re-allocate the 'msg_noa' pointer at a later time.
    msg_noa_offset = msg->header.packetSize - sizeof(bh_intp);

    //Stack of new base arrays
    std::stack<bh_base*> base_darys;

    for(int i=0; i<bhir->instr_list.size(); ++i)
    {
        const bh_instruction *inst = &bhir->instr_list[i];
        int nop = bh_operands_in_instruction(inst);
        for(bh_intp j=0; j<nop; ++j)
        {
            bh_view *operand = &bh_inst_operands((bh_instruction*)inst)[j];
            if(bh_is_constant(operand))
                continue;//No need to dispatch constants

            bh_base *base = bh_base_array(operand);
            // we assume that matrices are sent from the client
            // so for a matrix to be on the server, the client
            // must have sent it
            if(!ArrayMan_server_known_check(base))//The array is unknown to the server.
            {
                array_header *dary;
                res = ArrayMan_reserve_payload(sizeof(array_header),(void**) &dary);
                if(res != BH_SUCCESS)
                        return res;
                ArrayMan_server_known_insert(base);
                //The client memory pointer is the id of the array.
                dary->id = (bh_intp) base;
                dary->array = *base;
                ++noa;
                base_darys.push(base);
            }
        }
    }
    //Save the number of new arrays
    *((bh_intp*)((bh_intp)msg->data+(bh_intp)msg_noa_offset)) = noa;

    // at this point we are ready to send the data
    // first we send the bh_ir data stored in the buffer

    if(SendPacket(msg, filedes)!= BH_SRVR_SUCCESS)
    {
        printf("error in bhir send\n");
        return BH_ERROR;
    }
    // then we send all the matrices
    res = sendArrayData(base_darys, filedes);

    if(res != BH_SUCCESS)
        return res;

    // finally, inspect the bh_ir for sync's, discards and free's and handle them

    fd = filedes;
    res = bh_ir_map_instr(bhir, &bhir->dag_list[0], &inspect);

    if(res != BH_SUCCESS)
        return res;

    return BH_SUCCESS;
}
