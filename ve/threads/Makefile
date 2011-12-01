ROOT=../..
LIB=libcphvb_ve_mcore.so
CXX?=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -I$(ROOT)/include -I../score/ 
LDFLAGS=-L../../core -lcphvb -shared -Wl,-soname
SRC=cphvb_ve_mcore.c
OBJ=$(SRC:.c=.o)
INSTALLDIR?=/usr/local/lib

all: $(LIB) 

%.o: %.c $(ROOT)/include/*
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): cphvb_ve_mcore.o
		$(CXX) $< -o $@ $(LDFLAGS),$(LIB) $(CFLAGS)

clean:
		rm -rf *.o *~ *.so

install: all
		sudo cp $(LIB) $(INSTALLDIR)
