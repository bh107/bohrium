/*
 * PacketManagement.h
 *
 *  Created on: Feb 3, 2014
 *      Author: d
 */

#ifndef PACKETMANAGEMENT_H_
#define PACKETMANAGEMENT_H_
#include "Server_Error.h"
#include "ServerStructures.h"
#include "ArrayManagement.h"

void PacketMan_Init();
void PacketMan_Shutdown();

bh_server_error HandleSocket(socketListPtr ptr);

#endif /* PACKETMANAGEMENT_H_ */
