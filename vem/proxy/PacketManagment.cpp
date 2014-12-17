/*
 * PackageManagment.cpp
 *
 *  Created on: Feb 3, 2014
 *      Author: d
 */

#include <string.h>
#include <cassert>
#include <stdio.h>
#include <stack>
#include <bh_error.h>
#include <bh.h>
#include <StaticStore.hpp>
#include <cstring>

#include "PacketManagement.h"
#include "ServerStructures.h"
#include "BasicPacket.h"
#include "client_comp.h"


#define STRLENGTH 32 // assuming the last element to be nul terminator
#define DEFAULT_DATABUFFER_LENGTH 4;

// packet buffer
static bh_packet * packet;
static long bufferLength = DEFAULT_DATABUFFER_LENGTH;

// master array store
static StaticStore<bh_base> array_store(512);

// pointer to current pointer
static socketListPtr sockptr;

// function definitions
bh_server_error HandleCommand(bh_packet * packet, socketListPtr ptr);
bh_server_error sendresponce(packet_protocol ptc, bh_error error, int fd);
bh_base * add_new_server_array(bh_base * client_array, bh_intp client_id, socketListPtr ptr);
bool clientArrayIsKnown(bh_intp clientKey, socketListPtr ptr);
static bh_error inspect(bh_instruction *instr);

void PacketMan_Init()
{
    packet =  AllocatePacket(bufferLength);

}

void PacketMan_Shutdown()
{
    FreePacket(packet);
}



bh_server_error HandleSocket(socketListPtr ptr){

    //printf("Getting incoming packet\n");

    int res = ReceivePacket(ptr->socket_fd, packet, &bufferLength, DYNAMIC_BUFFER);
    if( res == BH_SRVR_SUCCESS )
    {
        //printf("Got a packet\n");
        res = HandleCommand(packet, ptr);
    }
    else if(res != BH_SRVR_SOCKET_CLOSED)
        printf("ReceivePacket failed\n");


    return res;
}

bh_server_error HandleCommand(bh_packet * packet, socketListPtr ptr)
{
    int bherr = BH_ERROR;
    switch (packet->header.packetID)
    {
    case BH_PTC_INIT:
        {
            //printf("INIT\n");
            char name[STRLENGTH];
            char * source = (char *)packet->data;
            strncpy(name, source, sizeof(name)-1);
            name[STRLENGTH-1] = '\0'; // nul terminate
            bherr = client_comp_init(name);
            //printf("Init complete\n");
            return sendresponce(BH_PTC_INIT, bherr, ptr->socket_fd);
        }
    case BH_PTC_SHUTDOWN:
        {
            //printf("SHUTDOWN\n");
            bherr = client_comp_shutdown();
            return sendresponce(BH_PTC_SHUTDOWN, bherr, ptr->socket_fd);
        }
    case BH_PTC_EXTMETHOD:
        {
            bh_opcode opcode = *((bh_opcode *)packet->data);
            char *name = (char*)packet->data+sizeof(bh_opcode);
            bherr = client_comp_extmethod(name, opcode);
            return sendresponce(BH_PTC_EXTMETHOD, bherr, ptr->socket_fd);
        }
    case BH_PTC_EXECUTE:
        {
            //printf("performing execute\n");

            //printdata("Bh-ir packet\0", packet->header.packetSize, packet->data);
            bh_ir bhir = bh_ir((char*)packet->data, packet->header.packetSize);

            // number of new arrays
            bh_intp *noa = (bh_intp*) ((bh_intp)(packet->data) + packet->header.packetSize);

            //array header list
            array_header * arys = (array_header*)(noa+1);

            std::stack<bh_base*> base_arys;
            bh_intp i=0;
            for(i=0; i < *noa; ++i)
            {
                bh_base *ary = add_new_server_array(&arys[i].array, arys[i].id, ptr);
                base_arys.push(ary);
            }

            // get the array data
            bh_packet dataPack;
            dataPack.data = NULL;

            while (!base_arys.empty())
            {
                bh_base *ary = base_arys.top();
                if(ary->data == NULL)
                {
                    base_arys.pop();
                    continue;
                }

                //set data to NULL so we can malloc space for data
                ary->data = NULL;
                bh_data_malloc(ary);
                dataPack.data = ary->data;
                // check if the data we got is aligned with the base
                int res = ReceiveArrayPacket(ptr->socket_fd, &dataPack, bh_base_size(ary), BH_PTC_ARRAY, (*ptr->map_server2client)[ary]);
                if(res == BH_SRVR_SUCCESS)
                {
                    //printdata("array packet", dataPack.header.packetSize, dataPack.data);
                    //with the base data now populated we can set the packet data pointer to NULL
                    dataPack.data = NULL;
                }
                else
                {
                    //if receive failed release the data
                    bh_data_free(ary);
                    dataPack.data = NULL;
                    ary->data = NULL;
                    if(res == BH_SRVR_PROTOCOLMISMATCH)
                    {
                        printf("Non array packet sent during array data assignment, id:%d\n", dataPack.header.packetID);
                    }
                    else if(res == BH_SRVR_BUFFERSIZE_ERR)
                    {
                        printf("Buffer size is not aligned with sent buffer size");
                    }
                    else if(res == BH_SRVR_ARRAYKEYMISMATCH)
                    {
                        printf("Error in array alignment\n");
                    }
                    else if(res == BH_SRVR_RECVERR)
                    {
                        printf("Receive failed when getting array data\n");
                    }
                }

                base_arys.pop();
            }

            // we now have all our bases on server memory space. we need to go through the bh_ir and change it
            // from referencing client addresses to server addresses

            for(uint64_t i=0; i < bhir.instr_list.size(); ++i)
            {
                bh_instruction *inst = &bhir.instr_list[i];
                int nop = bh_operands_in_instruction(inst);
                bh_view *ops = bh_inst_operands(inst);

                //Convert all instructon operands
                for(bh_intp j=0; j<nop; ++j)
                {
                    if(bh_is_constant(&ops[j]))
                        continue;
                    bh_base *base = bh_base_array(&ops[j]);
                    assert( clientArrayIsKnown((bh_intp)base, ptr));
                    bh_base_array(&ops[j]) = (*ptr->map_client2server)[(bh_intp)base];
                }
            }

            // with the bh_ir and matrices set we execute the bh_ir

            //bh_pprint_bhir(bhir);
            bherr = client_comp_execute(&bhir);

            if(bherr != BH_SUCCESS)
            {
                printf("Error from children->execute\n");
            }
            // finally, once the bh_ir is executed we need to handle all sync, free and discard commands
            else
            {
                //printf("Execution complete\n");
                //fflush (stdout);
                sockptr = ptr;
                for(uint64_t i=0; i < bhir.instr_list.size(); ++i)
                    bherr = inspect(&bhir.instr_list[i]);
                    if(bherr != BH_SUCCESS)
                        printf("map instruction err\n");
            }
            return sendresponce(BH_PTC_EXECUTE, bherr, ptr->socket_fd);
        }
    default:
        {
            printf("Unknown packet ID: %d\n", packet->header.packetID);
            return BH_SRVR_UNKNOWN;
        }
    }
}


bh_server_error sendresponce(packet_protocol ptc, bh_error error, int fd)
{
    bh_packet packet;
    memset(&packet, '0', sizeof(bh_packet));
    packet.header.packetID = ptc;
    packet.header.packetSize = sizeof(bh_error);
    packet.header.key = 0;
    packet.data = &error;
    return SendPacket(&packet, fd);
}

bool clientArrayIsKnown(bh_intp clientKey, socketListPtr ptr)
{
    return (*ptr->map_client2server).count(clientKey)>0;
}

bh_base * add_new_server_array(bh_base * client_array, bh_intp client_id, socketListPtr ptr)
{
    // get a new memory space for a base
    bh_base * array = array_store.c_next();

    // copy the base to the memory space
    *array = *client_array;

    assert((*ptr->map_client2server).count(client_id) == 0);
    assert((*ptr->map_server2client).count(array) == 0);
    (*ptr->map_client2server)[client_id] = array;
    (*ptr->map_server2client)[array]     = client_id;

    //printf("Client id: %p, server id: %p\n", (void*) client_id, array);

    return array;
}

void remove_known_server_array(bh_base * array)
{
    assert((*sockptr->map_server2client).count(array) == 1);

    bh_intp client_id = (*sockptr->map_server2client)[array];
    (*sockptr->map_server2client).erase(array);
    assert((*sockptr->map_client2server).count(client_id) == 1);
    (*sockptr->map_client2server).erase(client_id);
    array_store.erase(array);
}

static bh_error inspect(bh_instruction *instr)
{
    if(instr->opcode == BH_SYNC)
    {
//      printf("SYNC\n");
        bh_base *ary = bh_base_array(&instr->operand[0]);
        bh_packet pack;
        memset(&pack, '0', sizeof(bh_packet));
        pack.header.packetID = BH_PTC_SYNC;
        pack.header.packetSize = bh_base_size(ary);
        pack.header.key = (*sockptr->map_server2client)[ary];
        pack.data = ary->data;
        int res = SendPacket(&pack, sockptr->socket_fd);
        if(res == BH_SRVR_SUCCESS)
            return BH_SUCCESS;
        else
            return BH_ERROR;
    }
    else if(instr->opcode == BH_FREE)
    {
//      printf("FREE\n");
        bh_base *ary = bh_base_array(&instr->operand[0]);
        return bh_data_free(ary);
    }
    else if(instr->opcode == BH_DISCARD)
    {
//      printf("DISCARD\n");
        bh_base *ary = bh_base_array(&instr->operand[0]);
        if(ary->data!=NULL)
        {
            printf("Detected DISCARD without previous FREE\n");
            bh_data_free(ary);
        }
        remove_known_server_array(ary);
    }
    return BH_SUCCESS;
}


