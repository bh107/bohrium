ROOT=..
INCLUDE=../include
LIB=libcphvb.so
CC=gcc
WARNOPTS=-Wall
CFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=gnu99 -I$(ROOT)/include
LDFLAGS=-shared -Wl,-soname,$(LIB)
SRC=cphvb.c cphvb_opcode.c cphvb_type.c cphvb_error.c
OBJ=$(SRC:.c=.o)

all: $(LIB)

%.o: %.c $(INCLUDE)/*
		$(CC) -c $< -o $@ $(CFLAGS)

$(LIB): $(OBJ)
		$(CC) $(OBJ) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)
