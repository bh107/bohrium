ROOT=..
LIB=libbh.so
CC=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=c++98 -I$(ROOT)/include
LDFLAGS=-shared -ldl -Wl,-soname,$(LIB)
SRC=bh.c bh_opcode.c bh_type.c bh_error.c bh_component.c bh_pprint.c bh_memory.c bh_vcache.c bh_array.c bh_timing.c bh_vector.c
OBJ=$(SRC:.c=.o) $(ROOT)/core/bhir/*.o $(ROOT)/iniparser/libiniparser.a

INSTALLDIR?=/opt/bohrium/

all: $(LIB)

%.o: %.c $(ROOT)/include/* ../ve/shared/compute/*.hpp
	$(CC) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

clean:
	rm -rf *.o *~ $(LIB)

install: all
	cp $(LIB) $(INSTALLDIR)
