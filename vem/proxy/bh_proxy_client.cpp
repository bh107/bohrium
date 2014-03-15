#include <bh.h>

#include "Proxy_Client.h"
#include "Server_Error.h"
#include "CommandLine.h"

#define STDIN 0

// check if Key was pressed, need to have registered the stdio fd 0
int KeyPressed(fd_set * read_fds) {
	return FD_ISSET(STDIN, read_fds);
}

int main(int argc, char * argv[])
{
	bool running = true;

	char* address = NULL;
	int port = 0;

	if (argc == 5 &&                          \
        (strncmp(argv[1], "-a\0", 3) == 0) && \
        (strncmp(argv[3], "-p\0", 3) == 0)) {
		address = argv[2];
		port    = atoi(argv[4]);
	} else {
		printf("Usage: %s -a ipaddress -p port\n", argv[0]);
		return 0;
	}
	if (!address) {
		fprintf(stderr, "Please supply address.\n");
        return 0;
	}

	printf("Client will connect to [Address: %s, Port=%d].\n", address, port);

	// first initialize the data structure
	// then initialize the binding to the port
	if (Client_Init(port, 0) == BH_SRVR_SUCCESS && \
        Client_Start(address) == BH_SRVR_SUCCESS) {

		fd_set read_fds;
		// while the Client is running
		CommandLineInit();
		while(running) {
			// check all fds for new connections
			RunSelect(&read_fds);

			//handle new messages
			HandleConnections(&read_fds);

			if (KeyPressed(&read_fds)) {
				HandleCommand(&running);
			}

			if (!HasConnections()) {
				running = false;
			}
		}
		printf("Client shutdown\n");
	}

	Client_Shutdown();
	return 0;
}
