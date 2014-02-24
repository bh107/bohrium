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

static char def_port_no[6] = "3490\0";

int Client_Init(const char port_no[6], int * maxlq)
{
	if(clientInit == true)
	{
		return BH_SRVR_ALREADY_STARTED;
	}
	// clear client structure
	memset(&client, 0, sizeof(client));
	// initialize client with default values

	BhServerInit(&client);

	if(port_no!=NULL)
	{
		if (isNum(port_no, 6)) {
			int val = atoi(port_no);
			if(val >0 && val < MAX_PORT_NUMBER)
				strncpy(client.port_no, port_no, sizeof(client.port_no));
			else
			{
				printf("Port number %d invalid, defaulting to %s\n", val, def_port_no);
				strncpy(client.port_no, def_port_no, 6);
			}
		}
		else
		{
			printf("Unable to parse port number, defaulting to %s\n", def_port_no);
			strncpy(client.port_no, def_port_no, 6);
		}
	}
	else
		strncpy(client.port_no, def_port_no, 6);

	if(maxlq != NULL && *maxlq > 0 && *maxlq < MAX_LISTENING_QUEUE_SIZE)
		client.max_listening_queue = *maxlq;

	FD_SET(STDIN, &client.master_set);
	dataInit = true;


	PacketMan_Init();

	return BH_SRVR_SUCCESS;
}

int Client_Start(char * address)
{
	//make sure client data has been initialized
	if(dataInit==false)
	{
		return BH_SRVR_UNINITIALIZED;
	}
	//make sure the client doesn't try to startup again
	else if(clientInit==true)
		return BH_SRVR_ALREADY_STARTED;

	// file descriptors for the socket call and accepted connections
	int socket_fd;

	// Get information on client socket. Once info is gathered, attempt to
	// bind to first available setup
	struct addrinfo hints, *result, * ptr;

	hints = client.address_info;

	#ifdef PROXY_DEBUG
    printf("***Client initialization starting***\n\n");
	#endif

    //attempt to retrieve all relevant socket information
    int dwRetval = getaddrinfo(address, client.port_no, &hints, &result);
    if ( dwRetval != 0 ) {
        printf("getaddrinfo failed with error: %s\n", gai_strerror(dwRetval));
        return BH_SRVR_GETADDRINFO_ERR;
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
    	if(connect(socket_fd, ptr->ai_addr, ptr->ai_addrlen) == -1)
    	{
			close(socket_fd);
			printf("Connect failed with error: %s, continuing with next option\n", strerror(errno));
			continue;
    	}

    	break;
	}

    freeaddrinfo(result);
    // check if the connect was completed successfully
    if(ptr == NULL)
    {
    	printf("Client was unable to connect to port %s\n", client.port_no);
    	return BH_SRVR_SOCKET_BIND;
    }

    if(nodelay)
    {
		int flag = 1;
		int res = setsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY, (char *) &flag, sizeof(int));

		if(res < 0)
		{
			printf("Setsockopt failed with error: %s\n", strerror(errno));
			return BH_SRVR_ACCEPT_ERR;
		}
	}

	int res = PerformClientHandshake(socket_fd);

	if(res==BH_SRVR_SUCCESS)
	{
		#ifdef PROXY_DEBUG
		printf("Handshake completed successfully\n\n");
		#endif
		socketListPtr ptr = NewSockNode(&client);
		ptr->socket_fd = socket_fd;
		ptr->has_closed = false;
	    FD_SET(socket_fd, &client.master_set);
	    client.max_fd = socket_fd;
	}
	else
	{
		#ifdef PROXY_DEBUG
		printf("Handshake failed\n\n");
		#endif
		close(socket_fd);
	}
	printf("Client connect to port %s\n", client.port_no);
    clientInit = true;
	return BH_SRVR_SUCCESS;
}


int Client_Shutdown()
{
    printf("***Client shutdown started***\n\n");

    PacketMan_Shutdown();
    bool err = false;

    if(dataInit == true)
    {
    	// close the named socket
    	if( client.socket_file_descriptor != 0)
		{
			int val = close(client.socket_file_descriptor);
			if(val != 0)
			{
				printf("Closing socket file descriptor failed with error: %s\n", gai_strerror(val));
				err = true;
			}
		}

		// clear socket list
		if(CleanupSockets(&client)!=BH_SRVR_SUCCESS)
			err = true;
	}


	printf("***Client shutdown complete***\n\n");
	#ifdef PROXY_DEBUG
	fflush( stdout );
	#endif

	if(err == true)
		return BH_SRVR_SUCCESS;
	else
		return BH_SRVR_SHUTDOWN_ERR;
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
	if(clientInit==false || client.soc_list_start==NULL)
		return BH_SRVR_NULL;

	socketListPtr ptr;
	int res;
	// iterate through the socket list, checking if any connnections
	// need handling.

	for(ptr = client.soc_list_start;  ptr != NULL; ptr = ptr->next)
	{
		if(FD_ISSET(ptr->socket_fd, read_fds))
		{
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
