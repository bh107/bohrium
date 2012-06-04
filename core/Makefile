ROOT=..
LIB=libcphvb.so
CC=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=c++98 -I$(ROOT)/include
LDFLAGS=-shared -Wl,-soname,$(LIB)
SRC=cphvb.c cphvb_opcode.c cphvb_type.c cphvb_error.c cphvb_component.c cphvb_pprint.c cphvb_memory.c
OBJ=$(SRC:.c=.o)
INSTALLDIR?=/opt/cphvb/

all: $(LIB)

%.o: %.c $(ROOT)/include/*
		$(CC) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CC) $(OBJ) $(ROOT)/core/compute/cphvb_compute.o $(ROOT)/core/compute/cphvb_compute_random.o $(ROOT)/core/compute/cphvb_compute_reduce.o $(ROOT)/core/compute/cphvb_compute_matmul.o $(ROOT)/core/bundler/cphvb_bundler.o $(ROOT)/iniparser/libiniparser.a -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)

install: all
		sudo cp $(LIB) $(INSTALLDIR)
