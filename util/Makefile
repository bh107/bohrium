ROOT=..
INCLUDE=../include
LIB=libcphvbutil.so
CC=gcc
WARNOPTS=-Wall
CFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=gnu99 -I ../lib -I$(ROOT)/include
LDFLAGS=-shared -Wl,-soname,$(LIB) -L../lib
SRC=cphvbutil.c
OBJ=$(SRC:.c=.o)

all: $(LIB)

%.o: %.c $(INCLUDE)/*
		$(CC) -c $< -o $@ $(CFLAGS)

$(LIB): $(OBJ)
		$(CC) $(OBJ) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)
