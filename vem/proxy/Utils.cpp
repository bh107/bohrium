#include "Utils.h"
#include <arpa/inet.h>
#include <ctype.h>
#include <cstddef>
#include <cstdio>
#include <sys/socket.h>
#include <netdb.h>

// function for getting socket address
void *GetSInAddr(sockaddrptr ptr)
{
    if (ptr->sa_family == AF_INET)
        return &(((struct sockaddr_in*)ptr)->sin_addr);
    else if(ptr->sa_family == AF_INET6)
        return &(((struct sockaddr_in6*)ptr)->sin6_addr);
    else
        return NULL;
}

void PrintGetaddrinfoResults(struct addrinfo * result)
{

    struct addrinfo *ptr = NULL;
    struct sockaddr_in * si;
    int i = 1;

    printf("getaddrinfo response %d\n", i++);

    // Retrieve each address and print out the hex bytes
    for(ptr=result; ptr != NULL ;ptr=ptr->ai_next) {

        printf("getaddrinfo response %d\n", i++);
        printf("\tFlags: 0x%x\n", ptr->ai_flags);
        printf("\tFamily: ");
        switch (ptr->ai_family) {
            case AF_UNSPEC:
                printf("Unspecified\n");
                break;
            case AF_INET:
                printf("AF_INET (IPv4)\n");
                si = (struct sockaddr_in *) ptr->ai_addr;
                printf("\tIPv4 address %s\n",
                        inet_ntoa(si->sin_addr) );
                break;
            case AF_INET6:
                printf("AF_INET6 (IPv6) address\n");
                break;
            default:
                printf("Other %d\n", ptr->ai_family);
                break;
        }
        printf("\tSocket type: ");
        switch (ptr->ai_socktype) {
            case 0:
                printf("Unspecified\n");
                break;
            case SOCK_STREAM:
                printf("SOCK_STREAM (stream)\n");
                break;
            case SOCK_DGRAM:
                printf("SOCK_DGRAM (datagram) \n");
                break;
            case SOCK_RAW:
                printf("SOCK_RAW (raw) \n");
                break;
            case SOCK_RDM:
                printf("SOCK_RDM (reliable message datagram)\n");
                break;
            case SOCK_SEQPACKET:
                printf("SOCK_SEQPACKET (pseudo-stream packet)\n");
                break;
            default:
                printf("Other %d\n", ptr->ai_socktype);
                break;
        }
        printf("\tProtocol: ");
        switch (ptr->ai_protocol) {
            case 0:
                printf("Unspecified\n");
                break;
            case IPPROTO_TCP:
                printf("IPPROTO_TCP (TCP)\n");
                break;
            case IPPROTO_UDP:
                printf("IPPROTO_UDP (UDP) \n");
                break;
            default:
                printf("Other %d\n", ptr->ai_protocol);
                break;
        }
        printf("\tLength of this sockaddr: %d\n", ptr->ai_addrlen);
        printf("\tCanonical name: %s\n", ptr->ai_canonname);
    }
}

// print raw data
void printdata(char * name, long size, void * ptr)
{
    unsigned char* charPtr=(unsigned char*)ptr;
    int i;
    printf("structure name : %s\n", name);
    printf("structure size : %ld bytes\n", size);
    for(i=0;i<size;i++)
        printf("%02x ",charPtr[i]);
    fflush (stdout);
}

// function to check if string is nul terminated
// size is the size of the string including the nul terminator
// n+1

bool CheckNul(const char string[], int size)
{
    int i;
    for(i=0; i<size; i++)
    {
        if(string[i]=='\0')
            return true;
    }
    return false;
}

// checks if string is made up of numbers and is nul terminated
bool isNum(const char string[], int size)
{
    //at least the first char has to be a digit
    if(!(size > 0 && isdigit(string[0])))
        return false;

    int i;
    for(i=1; i<size; i++)
    {
        //if a char is not a digit it has to be nul
        if(!isdigit(string[i]))
        {
            if(string[i]=='\0')
                return true;
            else
                return false;
        }
    }
    //if we get here the string is made up of digits but is not nul terminated
    return false;
}
