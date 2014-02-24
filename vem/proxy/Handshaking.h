/*
 * Handshaking.h
 *
 *  Created on: Feb 6, 2014
 *      Author: d
 */

#ifndef HANDSHAKING_H_
#define HANDSHAKING_H_




// Function used by the server to verify connection with new clients
int PerformServerHandshake(int client_fd);

// Function used by clients to verify connection with the server
int PerformClientHandshake(int server_fd);


#endif /* HANDSHAKING_H_ */
