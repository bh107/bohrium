LIB=libsvi.so
CC=gcc
WARNOPTS=-Wall
CFLAGS= $(WARNOPTS) -fPIC -I../../lib -I../../util -std=gnu99
LDFLAGS=-shared -Wl,-soname,$(LIB)
SRC=cast.c free.c malloc.c read.c util.c add.c svi.c
OBJ=$(SRC:.c=.o)
DEFS=-D SVI_DEBUG

all: $(LIB)

%.o: %.c cast.h handler.h util.h svi.h
		$(CC) -c $< -o $@ $(CFLAGS) $(DEFS)

$(LIB): $(OBJ)
		$(CC) $(OBJ) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)
