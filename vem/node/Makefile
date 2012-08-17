ROOT=../..
LIB=libcphvb_vem_node.so
CC?=gcc
CXX?=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=gnu++0x -I$(ROOT)/include -I$(ROOT)/iniparser/src
LDFLAGS=-shared -L../../core -lcphvb -Wl,-soname,$(LIB)
SRC=cphvb_vem_node.cpp 
HEADER=cphvb_vem_node.h
OBJ=$(SRC:.cpp=.o)
INSTALLDIR?=/opt/cphvb/

all: $(LIB)

%.o: %.cpp $(HEADER)
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CXX) $(OBJ) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)

install: all
		cp $(LIB) $(INSTALLDIR)
