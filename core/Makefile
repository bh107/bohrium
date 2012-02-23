ROOT=..
LIB=libcphvb.so
CC?=gcc
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=gnu99 -I$(ROOT)/include
LDFLAGS=-shared -Wl,-soname,$(LIB)
SRC=cphvb.c cphvb_opcode.c cphvb_type.c cphvb_error.c cphvb_component.c cphvb_pprint.c
OBJ=$(SRC:.c=.o)
INSTALLDIR?=/opt/cphvb/

all: $(LIB)

%.o: %.c $(ROOT)/include/*
		$(CC) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CC) $(OBJ) $(ROOT)/iniparser/libiniparser.a -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)

install: all
		sudo cp $(LIB) $(INSTALLDIR)
