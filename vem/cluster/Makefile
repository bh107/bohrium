ROOT=../..
LIB=libcphvb_vem_cluster.so
CC=mpicc
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=gnu99 -I$(ROOT)/include -I$(ROOT)/iniparser/src
LDFLAGS=-shared -L../../core -lcphvb -Wl,-soname,$(LIB)
SRC=cphvb_vem_cluster.c process_grid.c
HEADER=cphvb_vem_cluster.h process_grid.h
OBJ=$(SRC:.c=.o)
INSTALLDIR?=/opt/cphvb/

all: $(LIB)

%.o: %.c $(HEADER)
		$(CC) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CC) $(OBJ) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)

install: all
		sudo cp $(LIB) $(INSTALLDIR)
