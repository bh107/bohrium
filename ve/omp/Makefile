ROOT=../..
LIB=libcphvb_ve_omp.so
CXX?=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -fopenmp -I$(ROOT)/include
LDFLAGS=-L../../core -lcphvb -lrt -lgomp -shared -Wl,-soname
SRC=cphvb_ve_omp.c
OBJ=$(SRC:.c=.o)
INSTALLDIR?=/usr/local/lib

all: $(LIB)

%.o: %.c $(ROOT)/include/* ./*.h ./*.hpp
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): cphvb_ve_omp.o
		$(CXX) $< -o $@ $(LDFLAGS),$(LIB) $(CFLAGS)

clean:
		rm -rf *.o *~ *.so

install: all
		sudo cp $(LIB) $(INSTALLDIR)
