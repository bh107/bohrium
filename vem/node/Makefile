ROOT=../..
INCLUDE=$(ROOT)/include
LIB=libcphvb_vem_node.so
CC=gcc
CXX=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=gnu++0x -I$(INCLUDE)
LDFLAGS=-shared -Wl,-soname,$(LIB)
SRC=cphvb_vem_node.cpp ArrayManager.cpp
HEADER=ArrayManager.hpp $(INCLUDE)/cphvb_vem_node.h
OBJ=$(SRC:.cpp=.o)
INSTALLDIR=/usr/local/lib

all: $(LIB)

%.o: %.cpp $(HEADER)
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CXX) $(OBJ) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)

install: all
		sudo cp $(LIB) $(INSTALLDIR)
