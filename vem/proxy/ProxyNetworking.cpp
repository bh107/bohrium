/*
 * ProxyNetworking.cpp
 *
 *  Created on: Feb 4, 2014
 *      Author: d
 */


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdbool.h>
#include <arpa/inet.h>
#include <bh.h>
#include <netinet/tcp.h>



#include "ProxyNetworking.h"
#include "ArrayManagement.h"
#include "Handshaking.h"
#include "BasicPacket.h"
#include "Server_Error.h"
#include "Utils.h"

#define MAX_PORT_NUMBER 65535
#define PORT "3490"
#define QUEUESIZE "10"
#define DEFAULT_DATABUFFER_LENGTH 1024

//data buffer for basic messages
bh_packet * recPack;
long bufferLength = DEFAULT_DATABUFFER_LENGTH;

static bool proxyInit = false;

static int  proxyfd;
static int  socket_fd;

bool no_delay = true;

// function definitions
bh_error Perform_Command(packet_protocol ptc, long packetSize, void * data);
bh_error GetResponse(packet_protocol ptc);

int Init_Networking(char * port_no)
{

    if(proxyInit == true)
        return BH_ERROR;
    // file descriptors for the socket call and accepted connections

    // Get information on server socket. Once info is gathered, attempt to
    // bind to first available setup
    struct addrinfo hints, *result, * ptr;

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    //attempt to retrieve all relevant socket information
    int dwRetval;
    if(port_no!=NULL && CheckNul(port_no, 6))
    {
        int val = atoi(port_no);
        if(val >0 && val < MAX_PORT_NUMBER)
        {
            dwRetval = getaddrinfo(NULL, port_no, &hints, &result);
        }
        else
            dwRetval = getaddrinfo(NULL, PORT, &hints, &result);
    }
    else
        dwRetval = getaddrinfo(NULL, PORT, &hints, &result);

    if ( dwRetval != 0 ) {
        printf("getaddrinfo failed with error: %s\n", gai_strerror(dwRetval));
        return BH_ERROR;
    }

    // print socket info
    #ifdef PROXY_DEBUG
    PrintGetaddrinfoResults(result);
    fflush( stdout );
    #endif

    int i = 0;
    // attempt to bind to the first relevant setting on target port
    for(ptr=result; ptr != NULL ;ptr=ptr->ai_next) {
        i++;
        socket_fd = socket(ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol);
        if(socket_fd == -1)
        {
            printf("Socket failed with error: %s, continuing with next option\n", strerror(errno));
            continue;
        }
        if(bind(socket_fd, ptr->ai_addr, ptr->ai_addrlen) == -1)
        {
            close(socket_fd);
            printf("Bind failed with error: %s, continuing with next option\n", strerror(errno));
            continue;
        }

        break;
    }

    freeaddrinfo(result);
    // check if the bind was completed successfully
    if(ptr == NULL)
    {
        printf("Server was unable to bind to port\n");
        return BH_SRVR_SOCKET_BIND;
    }

    if(no_delay)
    {
        int flag = 1;
        int res = setsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY, (char *) &flag, sizeof(int));

        if(res < 0)
        {
            printf("Setsockopt failed with error: %s\n", strerror(errno));
            return BH_SRVR_ACCEPT_ERR;
        }
    }

    printf("Server waiting for client to connect...\n");
    if(listen(socket_fd, 1) == -1)
    {
        printf("Listen failed with error: %s\n", strerror(errno));
        return BH_SRVR_LISTEN;
    }

    proxyfd = accept(socket_fd,
            (struct sockaddr *)NULL, NULL);

    if(proxyfd == -1)
    {
        printf("Accept failed with error: %s\n", strerror(errno));
        return BH_SRVR_ACCEPT_ERR;
    }

    int res = PerformServerHandshake(proxyfd);
    if(res != BH_SRVR_SUCCESS)
    {
        return res;
    }

    recPack =  AllocatePacket(DEFAULT_DATABUFFER_LENGTH);

    // init functions
    ArrayMan_init();

    #ifdef PROXY_DEBUG
    printf("***Proxy initialization completed***\n\n");
    #endif

    proxyInit = true;
    return BH_SUCCESS;
}

int Shutdown_Networking()
{
    FreePacket(recPack);

    //shutdown functions
    ArrayMan_shutdown();

    int val = close(socket_fd);
    if(val != 0)
    {
        printf("Closing socket file descriptor failed with error: %s\n", gai_strerror(val));
        return BH_ERROR;
    }
    val = close(proxyfd);
    if(val != 0)
    {
        printf("Closing socket file descriptor failed with error: %s\n", gai_strerror(val));
        return BH_ERROR;
    }
    return BH_SUCCESS;
}

bh_error nw_init(const char *component_name)
{
    return Perform_Command(BH_PTC_INIT, strlen(component_name) + 1, (void *) component_name);
}

bh_error nw_shutdown()
{
    return Perform_Command(BH_PTC_SHUTDOWN, 0, NULL);
}


bh_error nw_execute(bh_ir *bhir)
{
    //printf("NW EXE\n");
    int res = ArrayMan_client_bh_ir_package(bhir, proxyfd);
    if(res != BH_SUCCESS)
    {
        printf("bh_ir send failed");
        return res;
    }
    res = GetResponse(BH_PTC_EXECUTE);
    if(res!=BH_SUCCESS)
        printf("got error\n");
    return res;
}

bh_error nw_extmethod(const char *name, bh_opcode opcode)
{
    ArrayMan_reset_msg();
    ArrayMan_add_to_payload(sizeof(bh_opcode), &opcode);
    ArrayMan_add_to_payload(strlen(name)+1, (void*)name);
    ArrayMan_send_payload(BH_PTC_EXTMETHOD, proxyfd);
    return GetResponse(BH_PTC_EXTMETHOD);
}

bh_error GetResponse(packet_protocol ptc)
{

    int res = ReceivePacket(proxyfd, recPack, &bufferLength, DYNAMIC_BUFFER );

    if (res == BH_SRVR_SUCCESS) {
        if(recPack->header.packetID == ptc &&
           recPack->header.packetSize == sizeof(bh_error))
        {
            bh_error res = *(bh_error *)recPack->data;
            return res;
        }
        else if(recPack->header.packetID != ptc)
        {
            printf("Wrong packet id for response\n");
        }
        else
            printf("Wrong size of response data");
    }

    return BH_ERROR;
}
bh_error Perform_Command(packet_protocol ptc, long packetSize, void * data)
{
    bh_packet pack;
    pack.header.packetID = ptc;
    pack.header.packetSize = packetSize;
    pack.header.key = 0;
    pack.data = data;

    int res = SendPacket(&pack, proxyfd);
    if( res != BH_SRVR_SUCCESS)
        return BH_ERROR;

    return GetResponse(ptc);

}
