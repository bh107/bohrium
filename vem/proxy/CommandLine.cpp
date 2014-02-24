/*
 * CommandLine.cpp
 *
 *  Created on: Feb 14, 2014
 *      Author: d
 */

#include <cstdio>
#include <cstring>
#include <ctype.h>

#define BUFLEN 100

char buffer[BUFLEN];

void printHelpText()
{
	printf("Type '(q)uit' to stop client\n");
}

void CommandLineInit()
{
	printf("Client Started. ");
	printHelpText();
}

void clearbuffer()
{
	memset(buffer, ' ', BUFLEN);
}

bool HandleCommand(bool * running)
{

	if(fgets(buffer, BUFLEN, stdin) == NULL)
		return false;

	if(buffer[0]=='\n')
		return false;

	//find how large the string is
	int starttext = BUFLEN;
	int endl = 0;
	for(int i=0; i<BUFLEN; i++)
	{
		if(buffer[i]=='\n')
			break;

		if(endl < starttext && !isspace(buffer[i]))
			starttext = i;
		endl++;
	}

	// if there is not text in the buffer return false
	if(endl <= starttext)
		return false;

	if(((endl-starttext==4) && strncmp((buffer + starttext), "quit", 4) == 0) ||
	   ((endl-starttext==1) && strncmp((buffer + starttext), "q", 1)    == 0)  )
		*running = false;
	else
	{
		buffer[endl] = '\0';
		printf("Unknown command: %s\n", buffer);
		printHelpText();
	}


	return true;
}


