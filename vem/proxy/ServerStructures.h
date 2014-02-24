/*
 * ServerStructures.h
 *
 *  Created on: Jan 29, 2014
 *      Author: d
 */

#ifndef SERVERSTRUCTURES_H_
#define SERVERSTRUCTURES_H_


#include <bh.h>

#include <map>
#include <netdb.h>
#include <stdbool.h>


// singly linked list of sockets we are connected with

struct socketList
{
	int 		socket_fd;
	bool		has_closed;
	struct	sockaddr_storage targetAddress;
	struct 	socketList * next;
	// mapping of client to server and server to client addresses.
	// Because every socket is client specific we can safely limit
	// these maps here an ensure that the key value pairs are unique
	std::map<bh_intp, bh_base*> * map_client2server;
	std::map<bh_base*,bh_intp> * map_server2client;
} ;

typedef struct socketList socketList;
typedef socketList * socketListPtr;

// core structure for server

typedef struct
{
	char	 		port_no[6];
	int 			max_listening_queue;
	struct addrinfo address_info;
	int 			socket_file_descriptor;
	socketListPtr	soc_list_start;
	socketListPtr	soc_list_end;
	fd_set			master_set;
	int 			nodelay_option;
	int 			max_fd; //used for select()
} bh_server_info;

typedef bh_server_info * serverPtr;

#ifdef __cplusplus
extern "C" {
#endif
/*** Structure Functions ***/

// functions for bh_server_info

void BhServerInit(bh_server_info * info);

// functions for socket list

// initializes a new socket list node and adds it to the end
// of the sockets list passed as a parameter. If the server pointer
// is not specified the node is not created. this is to ensure
// that memory leaks don't happen. In this case a null pointer is
// returned. Since we are working on a singly linked list, elements
// are added to the end to preserve the data structure

socketListPtr NewSockNode(serverPtr server);

// remove nodes with file descriptors that have closed
int ClearClosedFDs(serverPtr server);

// clears all the sockets on the server
int CleanupSockets(serverPtr server);

#ifdef __cplusplus
}
#endif

#endif /* SERVERSTRUCTURES_H_ */
