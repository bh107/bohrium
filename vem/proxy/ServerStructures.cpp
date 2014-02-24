/*
 * ServerStructures.c
 *
 *  Created on: Jan 31, 2014
 *      Author: d
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netdb.h>

#include "ServerStructures.h"
#include "Server_Error.h"

void BhServerInit(bh_server_info * info)
{
	strncpy(info->port_no, "48576", sizeof(info->port_no)); // put something random
	info->max_listening_queue = 10;
	info->address_info.ai_family = AF_UNSPEC;
	info->address_info.ai_socktype = SOCK_STREAM;
	info->address_info.ai_flags = AI_PASSIVE;
	info->socket_file_descriptor = 0;
	info->nodelay_option = 1;
	info->soc_list_start= NULL;
	info->soc_list_end 	= NULL;
}

socketListPtr NewSockNode(serverPtr server)
{
	if(server==NULL)
		return NULL;

	socketListPtr newptr;

	//allocate memory for the new node.
	newptr = (socketListPtr) malloc(sizeof(socketList));
	newptr->next = 0;
	newptr->socket_fd = -1;
	// if the list hasn't been initialized set the start and end
	// to the new node
	if(server->soc_list_start==NULL)
	{
		//add the node to the list
		server->soc_list_start = newptr;
		//set the end to be the new node
		server->soc_list_end = newptr;
	}
	// otherwise add the node to the end and set the end to it
	else
	{
		//add node to the list
		server->soc_list_end->next = newptr;
		//point the end to it
		server->soc_list_end = newptr;
	}

	newptr->map_client2server = new std::map<bh_intp, bh_base*>();
	newptr->map_server2client = new std::map<bh_base*,bh_intp>();

	// return the new list element
	return newptr;
}

int ClearClosedFDs(serverPtr server)
{
	// if no nodes are on the list return
	if(server->soc_list_start == NULL)
		return BH_SRVR_SUCCESS;

	socketListPtr ptr, tmp, prev= NULL;
	ptr = server->soc_list_start;
	while( ptr != NULL)
	{
		// if the fd has closed
		if(ptr->has_closed == true)
		{
			printf("Clearing closed fd\n");
			// if this is the first element of the list
			if(prev == NULL)
			{
				server->soc_list_start = ptr->next;
				close(ptr->socket_fd);
				FD_CLR(ptr->socket_fd, &server->master_set);
				delete ptr->map_client2server;
				delete ptr->map_server2client;
				free(ptr);
				ptr = server->soc_list_start;
			}
			// otherwise
			else
			{
				tmp = ptr->next;
				close(ptr->socket_fd);
				delete ptr->map_client2server;
				delete ptr->map_server2client;
				free(ptr);
				prev->next = tmp;
				ptr = tmp;
			}
		}
		else
		{
			prev = ptr;
			ptr = ptr->next;
		}
	}

	return BH_SRVR_SUCCESS;
}

int CleanupSockets(serverPtr server)
{
	if(server == NULL)
	{
		printf("Server pointer is NULL in CleanupSockets\n");
		return BH_SRVR_NULL;
	}


	if(server->soc_list_start==NULL)
	{
		#ifdef SERVER_DEBUG
		printf("Socket List is empty\n");
		#endif
		return BH_SRVR_SUCCESS;
	}

	int err = BH_SRVR_SUCCESS;
	socketListPtr nextptr,ptr = server->soc_list_start;
	do
	{
		nextptr = ptr->next;
		int val = close(ptr->socket_fd);
		if(val != 0)
		{
			printf("Closing unnamed socket file descriptor failed with error: %s\n", gai_strerror(val));
			err = BH_SRVR_SHUTDOWN_ERR;
		}
		free(ptr);
		#ifdef SERVER_DEBUG
		//checking if end pointer is pointing at the end of the list
		if(nextptr == NULL)
		{
			if(ptr == server->soc_list_end)
				printf("End pointer list integrity check verified\n");
			else
				printf("End pointer not aligned to end of list."
						" Socket list corruption detected\n");
		}
		#endif
		ptr = nextptr;
	} while (ptr != NULL);

	return err;
}
