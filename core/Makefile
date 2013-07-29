ROOT=..
LIB=libbh.so
CC=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=c++98 -I$(ROOT)/include
LDFLAGS=-shared -ldl -Wl,-soname,$(LIB)
SRC=bh.c bh_opcode.c bh_type.c bh_error.c bh_component.c bh_pprint.c bh_memory.c bh_vcache.c bh_array.c bh_timing.c bh_dynamic_list.c
OBJ=$(SRC:.c=.o) $(ROOT)/core/compute/bh_compute.o $(ROOT)/core/compute/bh_compute_helpers.o $(ROOT)/core/compute/bh_compute_random.o $(ROOT)/core/compute/bh_compute_reduce.o $(ROOT)/core/compute/bh_compute_matmul.o $(ROOT)/core/compute/bh_compute_nselect.o $(ROOT)/core/compute/bh_typeutil.o $(ROOT)/core/bundler/bh_bundler.o $(ROOT)/core/graph/bh_graph.o $(ROOT)/core/bhir/bh_boolmat.o $(ROOT)/iniparser/libiniparser.a

INSTALLDIR?=/opt/bohrium/

all: $(LIB)

%.o: %.c $(ROOT)/include/* compute/*.hpp
		$(CC) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CC) $(OBJ) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)

install: all
		cp $(LIB) $(INSTALLDIR)
