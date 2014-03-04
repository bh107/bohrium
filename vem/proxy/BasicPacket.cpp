/*
 * SocketFunctions.c
 *
 *  Created on: Jan 31, 2014
 *      Author: d
 */

#include <string.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <stdio.h>
#include <errno.h>

#include "BasicPacket.h"
#include "Server_Error.h"
#include "timing.h"

//function definitions

int sendall(size_t msgsize, void * buffer, int file_descriptor);
int receiveall(size_t msgsize, void * buffer, int file_descriptor);



bh_server_error SendPacket(bh_packet * packet, int filedesc) {
    if (packet == NULL || filedesc < 0) {
        return BH_SRVR_SENDERR;
    }

    // send the packet header, after the header is sent we send the data
    int res = sendall(sizeof(packet_header), &packet->header, filedesc);
    if( res == BH_SRVR_SUCCESS)
    {
       return sendall(packet->header.packetSize, packet->data, filedesc);
    }
    return res;
}

bh_server_error ReceivePacket(int filedesc, bh_packet * packet, long * dataBufferSize, buffer_type flag) {

    if (filedesc < 0 || packet == NULL || (flag != CREATE_BUFFER && dataBufferSize == NULL)) {
        return BH_SRVR_RECVERR;
    }

    int res = receiveall(sizeof(packet_header), &packet->header, filedesc);
    if(res != BH_SRVR_SUCCESS)
    {
        if(res != BH_SRVR_SOCKET_CLOSED)
            printf("Socket closed\n");
        return res;
    }
    // check incoming data size
    if(packet->header.packetSize > 0 )
    {
        // if we have allocated memory and the packet size does not surpass the maximum
        if (flag == STATIC_BUFFER )
        {
            if( packet->data != NULL && *dataBufferSize > 0 && packet->header.packetSize <= *dataBufferSize )
            {
                res = receiveall(packet->header.packetSize, packet->data, filedesc);
                if(res != BH_SRVR_SUCCESS)
                {
                    printf("Error during packet data receive\n");
                    return res;
                }
            }
            else if (packet->header.packetSize > *dataBufferSize)
                        return BH_SRVR_BUFFERSIZE_ERR;
            else
                return BH_SRVR_RECVERR;
        }
        // if we have allocated memory an want to allow reallocation of any size to happen
        else if (flag == DYNAMIC_BUFFER )
        {
            if(!(*dataBufferSize >= 0))
            {
                return BH_SRVR_BUFFERSIZE_ERR;
            }
            else if (packet->data == NULL)
            {
                return BH_SRVR_RECVERR;
            }
            // if the packet we will receive is larger than the buffer
            if(packet->header.packetSize > *dataBufferSize)
            {
                packet->data = realloc(packet->data, packet->header.packetSize);
                if (packet->data == NULL) {
                    printf("Realloc failed during packet receive\n");
                    *dataBufferSize = 0;
                    return BH_SRVR_OUT_OF_MEMORY;
                }
                *dataBufferSize = packet->header.packetSize;
            }

            res = receiveall(packet->header.packetSize, packet->data, filedesc);
            if(res != BH_SRVR_SUCCESS)
            {
                printf("Error during packet data receive\n");
                return res;
            }
        }
        // if we wish to allow the function to perform the memory allocation
        else if (flag == CREATE_BUFFER )
        {
            if(packet->data != NULL)
            {
                printf("Non NULL data pointer passed in ReceivePacket with CREATE_BUFFER flag\n");
            }

            packet->data = malloc(packet->header.packetSize);
            if (packet->data == NULL) {
                printf("Malloc failed during packet receive: %s\n", strerror(errno));
                return BH_SRVR_OUT_OF_MEMORY;
            }
            if(dataBufferSize!=NULL)
                *dataBufferSize = packet->header.packetSize;
            res = receiveall(packet->header.packetSize, packet->data, filedesc);
            if(res != BH_SRVR_SUCCESS)
            {
                free(packet->data);
                printf("Error during packet data receive\n");
                return res;
            }
        }
        else
        {
            printf("Unknown flag in ReceivePacket\n");
            return BH_SRVR_RECVERR;
        }

    }
    else if(packet->header.packetSize < 0)
    {
        printf("Negative packet size in ReceivePacket\n");
        return BH_SRVR_RECVERR;
    }

    return BH_SRVR_SUCCESS;
}

// this version is used to populate a buffer with updated data. Thus it is more strict in data
// alignment and checks.
bh_server_error ReceiveArrayPacket(int filedesc, bh_packet * packet, long dataBufferSize, packet_protocol messageType, long keyVal) {

    if (filedesc < 0 || packet == NULL || packet->data == NULL) {
        return BH_SRVR_RECVERR;
    }

    int res = receiveall(sizeof(packet_header), &packet->header, filedesc);
    if(res != BH_SRVR_SUCCESS)
    {
        if(res != BH_SRVR_SOCKET_CLOSED)
            printf("Socket closed\n");
        return res;
    }
    // check incoming data size
    if( packet->data != NULL &&
        packet->header.packetSize == dataBufferSize &&
        messageType == packet->header.packetID &&
        packet->header.key == keyVal )
    {
        res = receiveall(packet->header.packetSize, packet->data, filedesc);
        if(res != BH_SRVR_SUCCESS)
        {
            printf("Error during packet data receive\n");
            return res;
        }
    }
    else if (messageType != packet->header.packetID)
        return BH_SRVR_PROTOCOLMISMATCH;
    else if (packet->header.key != keyVal)
        return BH_SRVR_ARRAYKEYMISMATCH;
    else if (packet->header.packetSize != dataBufferSize)
        return BH_SRVR_BUFFERSIZE_ERR;
    else
        return BH_SRVR_RECVERR;
    return BH_SRVR_SUCCESS;
}

// basic send and receive functions

bh_server_error sendall(size_t msgsize, void * buffer, int file_descriptor) {
    //
    if(msgsize == 0 || buffer == NULL)
        return BH_SRVR_SUCCESS;

    timing_sleep();

    size_t sendSize = msgsize;
    int sendlen;
    // pointer to the data buffer
    void * ptr = buffer;

    do {
        sendlen = send(file_descriptor, ptr, sendSize, 0);
        if (sendlen == 0) {
            //file descriptor has closed
            printf("Connection lost during send\n");
            return BH_SRVR_SOCKET_CLOSED;
        } else if (sendlen < 0) {
            // send error
            printf("Send error: %s\n", strerror(errno));
            return BH_SRVR_SENDERR;
        }

        //move pointer to point where data has not been sent
        ptr = (char*) ptr + sendlen;
        //reduce sent size
        sendSize -= sendlen;

    } while (sendSize > 0);

    if (sendSize != 0) {
        if (sendSize > 0) {
            printf("Not all data has been sent during sendall\n");
        } else {
            printf("Error during sendall\n");
        }
    }
    //after the header is sent we should handle any preprocessing and then send the data
    return BH_SRVR_SUCCESS;
}

bh_server_error receiveall(size_t msgsize, void * buffer, int file_descriptor) {
    size_t recvSize = msgsize;
    int recvlen;
    // pointer to the data buffer
    void * ptr = buffer;

    do {
        recvlen = recv(file_descriptor, ptr, recvSize, 0);
        if (recvlen == 0) {
            //file descriptor has closed
            if(recvSize == msgsize)
                printf("Connection closed\n");
            else
                printf("Connection closed during receiveall\n");
            return BH_SRVR_SOCKET_CLOSED;
        } else if (recvlen < 0) {
            // recv error
            printf("Recv error: %s\n", strerror(errno));
            return BH_SRVR_RECVERR;
        }

        //move pointer to point where data has not been sent
        ptr = (char*)ptr + recvlen;
        //reduce sent size
        recvSize -= recvlen;

    } while (recvSize > 0);

    if (recvSize != 0) {
        if (recvSize > 0) {
            printf("Data has not been received completely\n");
        } else {
            printf("Receive error in receiveall\n");
        }
    }

    timing_sleep();

    //after the header is sent we should handle any preprocessing and then send the data
    return BH_SRVR_SUCCESS;
}

bh_packet * AllocatePacket(long datasize)
{
    bh_packet * packet = (bh_packet*) malloc(sizeof(bh_packet));
    memset(packet, '0', sizeof(bh_packet));
    packet->header.packetSize = datasize;
    packet->header.packetID = BH_PTC_UNKNOWN;
    packet->header.key = 0;
    if(datasize > 0)
    {
        packet->data = malloc(packet->header.packetSize);
        memset(packet->data, '0', sizeof(packet->header.packetSize));
    }
    else
        packet->data = NULL;
    return packet;
}

void FreePacket(bh_packet * packet)
{
    free(packet->data);
    free(packet);
}


