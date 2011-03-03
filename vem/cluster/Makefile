ROOT=../..
INCLUDE=include
LIB=libcphvb_vem_cluster.so
CC=mpicc
WARNOPTS=-Wall
CFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=gnu99 -I$(ROOT)/$(INCLUDE)
LDFLAGS=-shared -Wl,-soname,$(LIB)
SRC=cphvb_vem_cluster.c
HEADER=private.h
OBJ=$(SRC:.c=.o)


all: $(LIB)

%.o: %.c $(ROOT)/$(INCLUDE)/* $(HEADER)
		$(CC) -c $< -o $@ $(CFLAGS)

$(LIB): $(OBJ)
		$(CC) $(OBJ) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)
