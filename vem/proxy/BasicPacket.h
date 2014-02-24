/*
 * SocketFunctions.h
 *
 *  Created on: Jan 30, 2014
 *      Author: d
 *
 * TODO:
 * Add function for predefined package sizes for every packet id
 * This would allow in ReceivePacket check the packet size if it
 * is correct and THEN perform malloc
 */

#ifndef SOCKETFUNCTIONS_H_
#define SOCKETFUNCTIONS_H_


#include "Server_Error.h"

#ifdef __cplusplus
extern "C" {
#endif

// Packet Structures

typedef enum
{
	BH_PTC_UNKNOWN = 0,
	BH_PTC_CLIENT_HANDSHAKE = 1,
	BH_PTC_SERVER_HANDSHAKE,
	BH_PTC_INIT,
	BH_PTC_SHUTDOWN,
	BH_PTC_EXTMETHOD,
	BH_PTC_EXECUTE,
	BH_PTC_ARRAY,
	BH_PTC_SYNC

} packet_protocol;


// Basic package structure

typedef struct
{
	packet_protocol packetID;
	long		packetSize;
	// used for array transfer to verify that
	// the data belongs to the corresponding array
	long		key;
} packet_header;


typedef struct
{
	packet_header header;
	void * data;
} bh_packet;

//recieve packet flags

typedef enum
{
	STATIC_BUFFER,
	DYNAMIC_BUFFER,
	CREATE_BUFFER
} buffer_type;

/*** Packet Functions ***/
/*
bh_packet BuildPacket(packet_IDs PacketType)
{

} */

// sends a packet to the specified firle descriptor
bh_server_error SendPacket(bh_packet * packet, int filedesc);


/*
 * Receives a packet from a file descriptor.
 * the buffer type flags define different functionality for the buffer
 *
 * -STATIC_BUFFER: The dataBufferSize is the maximum size, any packet that tries to surpass it will
 * generate a BH_SRVR_BUFFERSIZE_ERR
 *
 * -DYNAMIC_BUFFER: The data buffer will be reallocated if the size is too small for incoming packet data
 * dataBufferSize will have it's value changed to the new buffer size
 *
 * -CREATE_BUFFER: Will create a new memory buffer to fit the incoming package. It is expected that
 * the value of packet->data is NULL
 *
 * dataBufferSize for create buffer can be NULL. In all other cases it has to be a pointer to the size
 * of the data buffer.
 */
bh_server_error ReceivePacket( int filedesc, bh_packet * packet, long * dataBufferSize, buffer_type bt);

/*
 * Specialized receive used for receiving known arrays. Checks if all values match.
 *
 * If the message type doesn't match it returns a  BH_SRVR_PROTOCOLMISMATCH
 *
 * If the size of the buffer isn't equal with the data size it returns a BH_SRVR_BUFFERSIZE_ERR
 *
 * if the header keys don't match it returns a BH_SRVR_ARRAYKEYMISMATCH
 */

bh_server_error ReceiveArrayPacket(int filedesc, bh_packet * packet, long dataBufferSize, packet_protocol messageType, long keyVal);


//used to create basic packets
bh_packet * AllocatePacket(long datasize);

//used to free packets from ReceivePacket
void FreePacket(bh_packet * packet);





#ifdef __cplusplus
}
#endif

#endif /* SOCKETFUNCTIONS_H_ */
