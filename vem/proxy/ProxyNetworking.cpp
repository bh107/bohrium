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

#define DEFAULT_DATABUFFER_LENGTH 1024

//data buffer for basic messages
bh_packet * recPack;
long bufferLength = DEFAULT_DATABUFFER_LENGTH;

static bool proxyInit = false;
static int  proxyfd;

bool no_delay = true;
bool reuse_addr = true;

// function definitions
bh_error Perform_Command(packet_protocol ptc, long packetSize, void * data);
bh_error GetResponse(packet_protocol ptc);

int Init_Networking(uint16_t port)
{
    if (proxyInit == true) {
        return BH_SRVR_FAILURE;
    }

    //
    // Create a socket using reliable bi-directional communication.
    // This translates down to a TCP-based socket over ipv4.
    int socket_descriptor = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (-1 == socket_descriptor) {
        printf("Failed creating socket.\n");
        return BH_SRVR_FAILURE;
    }

    // 
    // Socket options
    int yes = 1;
    if (no_delay) {
        int sockopt_res = setsockopt(socket_descriptor, IPPROTO_TCP, TCP_NODELAY, (char *) &yes, sizeof(int));
        if (sockopt_res < 0) {
            fprintf(stderr,
                    "Setsockopt(TCP_NODELAY) failed with error: %s\n",
                    strerror(errno));
            return BH_ERROR;
        }
    }
    if (reuse_addr) {
        int sockopt_res = setsockopt(socket_descriptor, SOL_SOCKET, SO_REUSEADDR, (char *) &yes, sizeof(int));
        if (sockopt_res < 0) {
            fprintf(stderr,
                    "Setsockopt(SO_REUSEADDR) failed with error: %s\n",
                    strerror(errno));
            return BH_ERROR;
        }   
    }

    //
    // Setup data-structure for binding on the socket
    struct sockaddr_in server_addr;
    memset(&server_addr, '0', sizeof(server_addr));
    server_addr.sin_family         = AF_INET;
    server_addr.sin_addr.s_addr    = htonl(INADDR_ANY);
    server_addr.sin_port           = htons(port);

    int bind_res = bind(
        socket_descriptor, 
        (struct sockaddr*)&server_addr,
        sizeof(server_addr)
    );
    if (-1 == bind_res) {
        close(socket_descriptor);
        printf("Bind failed with error: [%s], exiting.\n", strerror(errno));
        return BH_SRVR_SOCKET_BIND;
    }

    //
    // Start listening for connections on the socket
    int listen_res = listen(socket_descriptor, 1);
    if (-1 == listen_res) {
        printf("Listen failed with error: %s\n", strerror(errno));
        return BH_SRVR_LISTEN;
    }

    #ifdef PROXY_DEBUG
    printf("Server waiting for client to connect...\n");
    #endif

    //
    // Block until a client connects
    proxyfd = accept(socket_descriptor, 0, 0);
    if (-1 == proxyfd) {
        printf("Accept failed with error: [%s]\n", strerror(errno));
        return BH_SRVR_ACCEPT_ERR;
    }

    //
    // Now close the socket since we only use a single incoming connection
    // in our lifetime...
    int close_res = close(socket_descriptor);
    if (-1 == close_res) {
        printf("Closing socket file descriptor failed with error: [%s]\n", strerror(errno));
        return BH_ERROR;
    }

    //
    // Start the "bohrium-proxy-protocol" handshake with client.
    int handshake_res = PerformServerHandshake(proxyfd);
    if (BH_SRVR_SUCCESS != handshake_res) {
        return handshake_res;
    }

    //
    // What is this?
    recPack = AllocatePacket(DEFAULT_DATABUFFER_LENGTH);

    // What is this? init functions?
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

    ArrayMan_shutdown();

    int close_res = close(proxyfd);
    if (-1 == close_res) {
        printf("Closing socket file descriptor failed with error [%s].", strerror(errno));
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
    int res = ArrayMan_client_bh_ir_package(bhir, proxyfd);
    if (BH_SUCCESS != res) {
        printf("bh_ir send failed");
        return res;
    }

    res = GetResponse(BH_PTC_EXECUTE);
    if (BH_SUCCESS != res) {
        printf("got error\n");
    }
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
           recPack->header.packetSize == sizeof(bh_error)) {
            bh_error res = *(bh_error *)recPack->data;
            return res;
        } else if(recPack->header.packetID != ptc) {
            printf("Wrong packet id for response\n");
        } else {
            printf("Wrong size of response data");
        }
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
    if (BH_SRVR_SUCCESS != res) {
        return BH_ERROR;
    }

    return GetResponse(ptc);

}
