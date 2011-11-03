ROOT=..
LIB=libcphvb.so
CC?=gcc
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=gnu99 -I$(ROOT)/include -I$(ROOT)/iniparser/src
LDFLAGS=-shared -Wl,-soname,$(LIB)
SRC=cphvb.c cphvb_opcode.c cphvb_type.c cphvb_error.c cphvb_conf.c
OBJ=$(SRC:.c=.o)
INSTALLDIR?=/usr/local/lib

all: $(LIB)

%.o: %.c $(ROOT)/include/* private.h
		$(CC) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CC) $(OBJ) $(ROOT)/iniparser/libiniparser.a -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)

install: all
		sudo cp $(LIB) $(INSTALLDIR)
