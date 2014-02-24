/*
 * bh_proxy_client.c
 *
 *  Created on: Jan 29, 2014
 *      Author: d
 */

#include <bh.h>

#include "Proxy_Client.h"
#include "Server_Error.h"
#include "CommandLine.h"

#define STDIN 0

// function definitions

// check if Key was pressed, need to have registered the stdio fd 0
int KeyPressed(fd_set * read_fds)
{
	return FD_ISSET(STDIN, read_fds);
}

int main(int argc, char * argv[])
{

	bool running = true;

	char * port = NULL;
	char * address = NULL;

	if(argc == 3 && (strncmp(argv[1], "-p\0", 3) == 0 || strncmp(argv[1], "-port\0", 6) == 0) )
	{
		port = argv[2];
	}
	else if(argc == 3 && (strncmp(argv[1], "-a\0", 3) == 0 || strncmp(argv[1], "-address\0", 9) == 0) )
	{
		address = argv[2];
	}
	else if(argc == 5 )
	{
		bool af = false;
		bool pf = false;
		if((strncmp(argv[1], "-a\0", 3) == 0 || strncmp(argv[1], "-address\0", 9) == 0) ) {
			address = argv[2];
			af = true;
		}
		else if((strncmp(argv[3], "-a\0", 3) == 0 || strncmp(argv[3], "-address\0", 9) == 0)) {
			address = argv[4];
			af = true;
		}

		if((strncmp(argv[1], "-p\0", 3) == 0 || strncmp(argv[1], "-port\0", 6) == 0) ) {
			port = argv[2];
			pf = true;
		}
		else if((strncmp(argv[3], "-p\0", 3) == 0 || strncmp(argv[3], "-port\0", 6) == 0)) {
			port = argv[4];
			pf = true;
		}

		if(!af || !pf)
		{
			printf("Usage: %s [-(p)ort port_number][-(a)ddress address_value]\n",argv[0]);
			return 0;
		}
	}
	else if(argc > 1)// || (argc == 2 && (strncmp(argv[1], "-h", 4) == 0 || strncmp(argv[1], "-help", 4) == 0 ) ) )
	{
		printf("Usage: %s [-(p)ort port_number][-(a)ddress address_value]\n",argv[0]);
		return 0;
	}

	printf("address: %s\n", address);

	// first initialize the data structure
	// then initialize the binding to the port
	if(Client_Init(port, 0) == BH_SRVR_SUCCESS &&
			Client_Start(address) == BH_SRVR_SUCCESS)
	{
		fd_set read_fds;
		// while the Client is running
		CommandLineInit();
		while(running)
		{
			// check all fds for new connections
			RunSelect(&read_fds);

			//handle new messages
			HandleConnections(&read_fds);

			if(KeyPressed(&read_fds))
				HandleCommand(&running);

			if(!HasConnections())
				running = false;
		}
		printf("Client shutdown\n");
	}

	Client_Shutdown();
	return 0;
}
