/*
 * Client.h
 *
 *  Created on: Jan 29, 2014
 *      Author: d
 */

#ifndef PROXY_CLIENT_H_
#define PROXY_CLIENT_H_

//#define PROXY_DEBUG

#ifdef __cplusplus
extern "C" {
#endif

//#define PROXY_DEBUG

// Initializes the Client
// if Client has started will return an already started signal
int Client_Init(const char * port_no, int * maxconnect);

// starts the Client
// will return a data not initialied exception if Client_Init_Data
// has not been performed. Will also return an already started signal
// if this function has been run before

int Client_Start(char* address);

// Causes the Client to block while it waits for a client to connect
// Will accept one new client and register it. Will fail if Client
// has not been initialized
int Establish_Connection();

// Terminates all connections and performs cleanup of all internal
// data structures
int Client_Shutdown();

// Functions for I/O multiplexing
// perfoms a select() on the sockets the Client is connected with
int RunSelect(fd_set * read_fds);

// checks if the Client has a new connection
int HasNewConnection(fd_set * read_fds);

// handles all messages that have been sent by clients.
int HandleConnections(fd_set * read_fds);

// checks if there are socket connections for handling
bool HasConnections();

#ifdef __cplusplus
}
#endif
#endif /* PROXY_H_ */
