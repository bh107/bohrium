/*
 * Proxy_Client.cpp
 *
 *  Created on: Jan 29, 2014
 *      Author: d
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdbool.h>
#include <netinet/tcp.h>

#include "Proxy_Client.h"
#include "Server_Error.h"
#include "ServerStructures.h"
#include "BasicPacket.h"
#include "Utils.h"
#include "Handshaking.h"
#include "PacketManagement.h"

#define MAX_PORT_NUMBER 65535
#define MAX_LISTENING_QUEUE_SIZE 100
#define STDIN 0

static bh_server_info client;
static bool dataInit = false;
static bool clientInit = false;

bool nodelay = true;

/**
 *  Must supply a valid port.
 */
int Client_Init(uint16_t port, int * maxlq)
{
	if (true == clientInit) {
		return BH_SRVR_ALREADY_STARTED;
	}

	// clear client structure
	memset(&client, 0, sizeof(client));
	// initialize client with default values

	BhServerInit(&client);
    client.port = port;

	if (maxlq != NULL && *maxlq > 0 && *maxlq < MAX_LISTENING_QUEUE_SIZE) {
		client.max_listening_queue = *maxlq;
    }

	FD_SET(STDIN, &client.master_set);
	dataInit = true;

	PacketMan_Init();

	return BH_SRVR_SUCCESS;
}

int Client_Start(const char* ipaddr)
{
	if (dataInit==false) {          // Ensure client initialization
		return BH_SRVR_UNINITIALIZED;
	} else if(clientInit==true) {   // Ensure client doesn't try to startup again
		return BH_SRVR_ALREADY_STARTED;
    }

    //
    // TODO: Add support for using a hostname
    
    //
    // Setup address struct describing the server we want to connect to
    struct sockaddr_in server_addr;
    memset(&server_addr, '0', sizeof(server_addr));
    server_addr.sin_family  = AF_INET;
    server_addr.sin_port    = htons(client.port);

    //
    // Convert ip-address to binary form
    int s = inet_pton(AF_INET, ipaddr, &server_addr.sin_addr);
    if (0 >= s) {
        if (0 == s) {
           fprintf(stderr, "Not in presentation format");
        } else {
           fprintf(stderr, "Invalid server-ip: %s\n", strerror(errno));
           return BH_SRVR_FAILURE;
        }
    }

    //
	// Create socket
	int socket_descriptor = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (-1 == socket_descriptor) {
        fprintf(stderr,
                "Failed creating socket got error: %s\n",
                strerror(errno));
        return BH_SRVR_FAILURE;
    }

    // 
    // Set socket options
    if (nodelay) {
        int flag = 1;
        int res = setsockopt(socket_descriptor, IPPROTO_TCP, TCP_NODELAY, (char *) &flag, sizeof(int));

        if(res < 0) {
            fprintf(stderr,
                    "Setsockopt failed with error: %s\n",
                    strerror(errno));
            return BH_SRVR_ACCEPT_ERR;
        }
    }

    //
    // Attempt to connect to the server
    int connect_res = connect(socket_descriptor, 
                              (struct sockaddr *)&server_addr, 
                              sizeof(server_addr));
    if (-1 == connect_res) {
        fprintf(stderr,
                "Failed connecting to server, got error: %s\n",
                strerror(errno));
        return BH_SRVR_FAILURE;
    }

	int handshake_res = PerformClientHandshake(socket_descriptor);

	if (BH_SRVR_SUCCESS==handshake_res) {
		#ifdef PROXY_DEBUG
		fprintf(stdout, "Handshake completed successfully\n\n");
		#endif
		socketListPtr ptr = NewSockNode(&client);
		ptr->socket_fd = socket_descriptor;
		ptr->has_closed = false;
	    FD_SET(socket_descriptor, &client.master_set);
	    client.max_fd = socket_descriptor;
	} else {
		#ifdef PROXY_DEBUG
		fprintf(stderr, "Handshake failed\n\n");
		#endif
		close(socket_descriptor);
	}
    #ifdef PROXY_DEBUG
	fprintf(stdout, "Client connected to port %s\n", client.port_no);
    #endif
    clientInit = true;
	return BH_SRVR_SUCCESS;
}

int Client_Shutdown()
{
	#ifdef PROXY_DEBUG
    printf("***Client shutdown started***\n\n");
	#endif

    PacketMan_Shutdown();
    bool err = false;

    if (dataInit == true) {
    	// close the named socket
    	if (client.socket_file_descriptor != 0) {
			int val = close(client.socket_file_descriptor);
			if (val != 0) {
				printf("Closing socket file descriptor failed with error: %s\n", gai_strerror(val));
				err = true;
			}
		}

		// clear socket list
		if (BH_SRVR_SUCCESS != CleanupSockets(&client)) {
			err = true;
        }
	}

	#ifdef PROXY_DEBUG
	printf("***Client shutdown complete***\n\n");
	fflush( stdout );
	#endif

	if (err == true) {
		return BH_SRVR_SUCCESS;
    } else {
		return BH_SRVR_SHUTDOWN_ERR;
    }
}

// used in collaboration with IsNewConnection
// for I/O multiplexing
int RunSelect(fd_set * read_fds)
{
	*read_fds = client.master_set;
	if (select(client.max_fd+1, read_fds, NULL, NULL, NULL) == -1) {
		printf("RunSelect failed with error: %s\n", strerror(errno));
		return BH_SRVR_SELECTERR;
	}
	return BH_SRVR_SUCCESS;
}

int HasNewConnection(fd_set * read_fds)
{
	return FD_ISSET(client.socket_file_descriptor, read_fds);
}

bool HasConnections()
{
	return (client.soc_list_start != NULL);
}

// check if there is a new connection
int HandleConnections(fd_set * read_fds)
{
	//if there are not connections or the client hasn't started
	// return NULL
	if (clientInit==false || client.soc_list_start==NULL) {
		return BH_SRVR_NULL;
    }

	socketListPtr ptr;
	int res;
	// iterate through the socket list, checking if any connnections
	// need handling.

	for(ptr = client.soc_list_start;  ptr != NULL; ptr = ptr->next) {
		if(FD_ISSET(ptr->socket_fd, read_fds)) {
			#ifdef PROXY_DEBUG
			printf("Handling new request\n");
			fflush( stdout );
			#endif
			res = HandleSocket(ptr);
			if(res == BH_SRVR_SOCKET_CLOSED)
				ptr->has_closed = true;
			#ifdef PROXY_DEBUG
			printf("\n");
			fflush( stdout );
			#endif
		}
	}

	// close all connections that failed or where determined as closed
	ClearClosedFDs(&client);
	return BH_SRVR_SUCCESS;
}
