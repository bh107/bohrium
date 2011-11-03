ROOT=../..
LIB=libcphvb_ve_score.so
CXX?=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -I$(ROOT)/include
LDFLAGS=-L../../core -lcphvb -shared -Wl,-soname
SRC=cphvb_ve_score.c
OBJ=$(SRC:.c=.o)
INSTALLDIR=/usr/local/lib

all: libcphvb_ve_mcore.so libcphvb_ve_score.so

%.o: %.c $(ROOT)/include/*
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

libcphvb_ve_mcore.so: cphvb_ve_mcore.o
		$(CXX) cphvb_ve_mcore.o -o $@ $(LDFLAGS),libcphvb_ve_mcore.so $(CFLAGS)

libcphvb_ve_score.so: cphvb_ve_score.o
		$(CXX) cphvb_ve_score.o -o $@ $(LDFLAGS),libcphvb_ve_score.so $(CFLAGS)

clean:
		rm -rf *.o *~ *.so
