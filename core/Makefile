ROOT=..
include $(ROOT)/platform.mk
LIB=libbh.$(LIB_SUFFIX)
CC=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) $(ARCH_OPTS) -g -O2 -fPIC -std=c++98 -I$(ROOT)/include
LDFLAGS= $(ARCH_OPTS) -shared -ldl $(EXTRA_LIBS) -Wl,$(LD_LIB_COMMAND),$(LIB)
SRC=$(ALL_C_FILES)
OBJ=$(SRC:.c=.o) $(ROOT)/core/bhir/*.o $(ROOT)/iniparser/libiniparser.a

all: $(LIB)

%.o: %.c $(ROOT)/include/*
	$(CC) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

clean:
	rm -rf *.o *~ $(LIB)

install: all
	cp $(LIB) $(INSTALLDIR)
