ROOT=..
LIB=libcphvb.so
CC=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=c++98 -I$(ROOT)/include
LDFLAGS=-shared -Wl,-soname,$(LIB)
SRC=cphvb.c cphvb_opcode.c cphvb_type.c cphvb_error.c cphvb_component.c cphvb_pprint.c cphvb_memory.c cphvb_vcache.c cphvb_array.c
OBJ=$(SRC:.c=.o) $(ROOT)/core/compute/cphvb_compute.o $(ROOT)/core/compute/cphvb_compute_helpers.o $(ROOT)/core/compute/cphvb_compute_random.o $(ROOT)/core/compute/cphvb_compute_reduce.o $(ROOT)/core/compute/cphvb_compute_matmul.o $(ROOT)/core/compute/cphvb_compute_nselect.o $(ROOT)/core/bundler/cphvb_bundler.o $(ROOT)/iniparser/libiniparser.a
INSTALLDIR?=/opt/cphvb/

all: $(LIB)

%.o: %.c $(ROOT)/include/* compute/*.hpp
		$(CC) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CC) $(OBJ) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)

install: all
		cp $(LIB) $(INSTALLDIR)
