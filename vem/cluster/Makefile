ROOT=../..
INCLUDE=include
LIB=libcphvb_vem_cluster.so
CC=mpicc
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=gnu99 -I$(ROOT)/$(INCLUDE)
LDFLAGS=-shared -Wl,-soname,$(LIB)
SRC=cphvb_vem_cluster.c handle.c
HEADER=private.h handle.h message.h
OBJ=$(SRC:.c=.o)
INSTALLDIR=/usr/local/lib

all: $(LIB)

%.o: %.c $(ROOT)/$(INCLUDE)/* $(HEADER)
		$(CC) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CC) $(OBJ) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)

install: all
		sudo cp $(LIB) $(INSTALLDIR)
