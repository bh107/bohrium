CC=gcc
WARNOPTS= -Wall
CFLAGS= $(WARNOPTS) -g -O2 -I../../../lib -I../../../util -I.. -std=gnu99 
LDFLAGS=-L../../../lib -L../../../util -L.. -lcphvb -lcphvbutil -lsvi
SRC=svi-test.c
OBJ=$(SRC:.c=.o)

all: svi-test

%.o: %.c
		$(CC) -c $< -o $@ $(CFLAGS)


svi-test: svi-test.o
		$(CC) $< -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ svi-test
