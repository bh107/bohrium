ROOT=..
INCLUDE=../include
LIB=libcphvem.so
CC=gcc
WARNOPTS=-Wall
CFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=gnu99 -I$(ROOT)/include
LDFLAGS=-shared -Wl,-soname,$(LIB)
SRC=cphvem.c
HEADER=private.h
OBJ=$(SRC:.c=.o)


all: $(LIB)

%.o: %.c $(INCLUDE)/* $(HEADER)
		$(CC) -c $< -o $@ $(CFLAGS)

$(LIB): $(OBJ)
		$(CC) $(OBJ) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)
