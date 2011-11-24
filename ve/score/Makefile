ROOT=../..
LIB=libcphvb_ve_score.so
CXX?=g++
WARNOPTS=-Wall
#LCFLAGS= $(WARNOPTS) -g -O3 -fPIC -I$(ROOT)/include
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -I$(ROOT)/include
#LCFLAGS= $(WARNOPTS) -g -fPIC -I$(ROOT)/include
LDFLAGS=-L../../core -lcphvb -lrt -shared -Wl,-soname
SRC=cphvb_ve_score.c
OBJ=$(SRC:.c=.o)
INSTALLDIR?=/usr/local/lib

all: $(LIB) 

%.o: %.c $(ROOT)/include/*
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): cphvb_ve_score.o
		$(CXX) $< -o $@ $(LDFLAGS),$(LIB) $(CFLAGS)

clean:
		rm -rf *.o *~ *.so

install: all
		sudo cp $(LIB) $(INSTALLDIR)
