ROOT=../..
LIB=libcphvb_ve_simple.so
CC=gcc
WARNOPTS=-Wall
CFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=gnu99 -I$(ROOT)/include
LDFLAGS=-shared -Wl,-soname,$(LIB)
SRC=cphvb_ve_simple.c
OBJ=$(SRC:.c=.o)

all: $(LIB)

%.o: %.c $(ROOT)/include/*
		$(CC) -c $< -o $@ $(CFLAGS)

$(LIB): $(OBJ)
		$(CC) $(OBJ) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)
