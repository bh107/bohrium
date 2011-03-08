ROOT=../..
LIB=libcphvb_ve_simple.so
CC=gcc
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=gnu99 -I$(ROOT)/include
LDFLAGS=-shared -Wl,-soname,$(LIB)
SRC=cphvb_ve_simple.c
OBJ=$(SRC:.c=.o)
INSTALLDIR=/usr/local/lib

all: $(LIB)

%.o: %.c $(ROOT)/include/*
		$(CC) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CC) $(OBJ) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)

install: all
		sudo cp $(LIB) $(INSTALLDIR)
