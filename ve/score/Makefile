ROOT=../..
LIB=libcphvb_ve_score.so
CC=g++
WARNOPTS=-Wall
CFLAGS=-lc -lm -pthread -lcphvb
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -I$(ROOT)/include
LDFLAGS=-shared -Wl,-soname
SRC=cphvb_ve_score.c
OBJ=$(SRC:.c=.o)
INSTALLDIR=/usr/local/lib

all: libcphvb_ve_mcore.so libcphvb_ve_score.so 

%.o: %.c $(ROOT)/include/*
		$(CC) -c $< $(CFLAGS) -o $@ $(LCFLAGS) $(CFLAGS)

libcphvb_ve_mcore.so: cphvb_ve_mcore.o
		$(CC) cphvb_ve_mcore.o -o $@ $(LDFLAGS),libcphvb_ve_mcore.so $(CFLAGS)

libcphvb_ve_score.so: cphvb_ve_score.o
		$(CC) cphvb_ve_score.o -o $@ $(LDFLAGS),libcphvb_ve_score.so $(CFLAGS)

clean:
		rm -rf *.o *~ *.so

install: all
		sudo cp libcphvb_ve_score.so $(INSTALLDIR)
		sudo cp libcphvb_ve_mcore.so $(INSTALLDIR)
