ROOT=../..
LIB=libcphvb_ve_score.so
CXX?=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -I$(ROOT)/include
LDFLAGS=-L../../core -lcphvb -lrt -shared -Wl,-soname
SRC=cphvb_ve_score.c bundler.cpp
HEADER=cphvb_ve_score.h bundler.hpp traverse.hpp $(ROOT)/include/*
OBJTMP=$(SRC:.c=.o)
OBJ=$(OBJTMP:.cpp=.o)
INSTALLDIR?=/usr/local/lib

all: $(LIB)

%.o: %.c $(HEADER)
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

%.o: %.cpp $(HEADER)
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CXX) $(OBJ) -o $@ $(LDFLAGS),$(LIB) $(CFLAGS)

clean:
		rm -rf *.o *~ *.so

install: all
		sudo cp $(LIB) $(INSTALLDIR)

