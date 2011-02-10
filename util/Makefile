LIB=libcphvbutil.so
CC=gcc
WARNOPTS=-Wall
CFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=gnu99 -I ../lib
LDFLAGS=-shared -Wl,-soname,$(LIB) -L../lib
SRC=cphvbutil.c
HEADERS= cphvbutil.h
OBJ=$(SRC:.c=.o)

all: $(LIB)

%.o: %.c $(HEADERS)
		$(CC) -c $< -o $@ $(CFLAGS)

$(LIB): $(OBJ)
		$(CC) $(OBJ) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)
