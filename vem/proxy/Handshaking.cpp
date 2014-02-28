/*
 * Handshaking.cpp
 *
 *  Created on: Feb 6, 2014
 *      Author: d
 */

#include <string.h>
#include <stdio.h>
#include <errno.h>

#include "BasicPacket.h"
#include "Server_Error.h"

// basic handshaking functions

//server and client handshake "keys"
static char serverHandshake[] = "BhServerHandshake";
static char clientHandshake[] = "BhClientHandshake";

int PerformServerHandshake(int client_fd) {
#ifdef SERVER_DEBUG
	printf("Starting handshake, waiting for client...\n");
	fflush( stdout );
#endif

	int res;
	long packlen = strlen(clientHandshake);
	bh_packet * recPack = AllocatePacket(packlen);
	res = ReceivePacket(client_fd, recPack, &packlen, STATIC_BUFFER);
	if (res == BH_SRVR_SUCCESS) {
		if (strncmp(clientHandshake, (char *) recPack->data,
				strlen(clientHandshake)) == 0) {
#ifdef SERVER_DEBUG
			printf("%s\n", (char *) recPack->data);
			printf("client handshake recieved, sending handshake...\n");
			fflush( stdout );
#endif
		} else {
#ifdef SERVER_DEBUG
			printf("client handshake key incorrect\n");
#endif

			FreePacket(recPack);
			return BH_SRVR_HANDSHAKE_KEY_MISMATCH;
		}
	} else {
#ifdef SERVER_DEBUG
		printf("client handshake failed\n");
#endif
		return res;
	}
	FreePacket(recPack);
	// if we got the correct handshake from the client, respond
	bh_packet pack;
	memset(&pack, 0, sizeof(bh_packet));
	pack.header.packetID = BH_PTC_SERVER_HANDSHAKE;
	pack.header.packetSize = strlen(serverHandshake);
	pack.data = serverHandshake;
	SendPacket(&pack, client_fd);
#ifdef SERVER_DEBUG
	printf("handshake sent\n");
	fflush( stdout );
#endif
	return BH_SRVR_SUCCESS;
}


int PerformClientHandshake(int server_fd) {
#ifdef SERVER_DEBUG
	printf("Starting handshake, sending handshake to server...\n");
	fflush( stdout );
#endif

	bh_packet pack;
	memset(&pack, 0, sizeof(bh_packet));
	pack.header.packetID = BH_PTC_CLIENT_HANDSHAKE;
	pack.header.packetSize = strlen(clientHandshake);
	pack.data = clientHandshake;
	SendPacket(&pack, server_fd);
#ifdef SERVER_DEBUG
	printf("handshake sent, waiting for server responce...\n");
	fflush( stdout );
#endif

	int res;
	long packlen = strlen(serverHandshake);
	bh_packet * recPack =  AllocatePacket(packlen);
	res = ReceivePacket(server_fd, recPack, &packlen, STATIC_BUFFER);
	if (res == BH_SRVR_SUCCESS) {
		if (strncmp(serverHandshake, (char *) recPack->data,
				strlen(serverHandshake)) == 0) {
#ifdef SERVER_DEBUG
			printf("server handshake recieved\n");
			fflush( stdout );
#endif
		} else {
#ifdef SERVER_DEBUG
			printf("server handshake key incorrect\n");
#endif
			FreePacket(recPack);
			return BH_SRVR_HANDSHAKE_KEY_MISMATCH;
		}
	} else {
#ifdef SERVER_DEBUG
		printf("server handshake failed\n");
#endif
		return res;
	}

	// if we got the correct handshake from the client, respond

	FreePacket(recPack);
	return BH_SRVR_SUCCESS;
}

