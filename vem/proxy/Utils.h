/*
 * Utls.h
 *
 *  Created on: Jan 29, 2014
 *      Author: d
 */

#ifndef UTLS_H_
#define UTLS_H_

#include <arpa/inet.h>
#include <ctype.h>


typedef struct sockaddr * sockaddrptr;

// function for getting socket address
void *GetSInAddr(sockaddrptr ptr);

void PrintGetaddrinfoResults(struct addrinfo * result);

// print raw data
void printdata(char * name, long size, void * ptr);

// function to check if string is nul terminated
// size is the size of the string including the nul terminator
// n+1

bool CheckNul(const char string[], int size);

// checks if string is made up of numbers and is nul terminated
bool isNum(const char string[], int size);


#endif /* UTLS_H_ */
