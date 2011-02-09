LIB=libcphvb.so
CC=gcc
WARNOPTS=-Wall
CFLAGS= $(WARNOPTS) -O2 -fPIC -std=gnu99
LDFLAGS=-shared -Wl,-soname,$(LIB)
SRC=cphvb.c snprint.c util.c opcode.c type.c error.c
HEADERS= cphvb.h error.h opcode.h type.h error.h
OBJ=$(SRC:.c=.o)

all: $(LIB)

%.o: %.c $(HEADERS)
		$(CC) -c $< -o $@ $(CFLAGS)

$(LIB): $(OBJ)
		$(CC) $(OBJ) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)
