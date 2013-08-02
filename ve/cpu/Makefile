ROOT=../..
LIB=libbh_ve_cpu.so
CXX?=g++
WARNOPTS=-Wall
EXTRAS+=
TIMING+=
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -I$(ROOT)/include -I$(ROOT)/core -I$(ROOT)/traverse $(EXTRAS)
LDFLAGS=-L../../core -lbh -lrt -shared -Wl,-soname
HEADER=./*.h $(ROOT)/include/*
SRC= $(shell find -iname '*.cpp') $(shell find -iname '*.c')
OBJTMP=$(SRC:.c=.o)
OBJ=$(OBJTMP:.cpp=.o) $(ROOT)/ve/shared/compute/bh_compute.o $(ROOT)/ve/shared/compute/bh_compute_helpers.o $(ROOT)/ve/shared/compute/bh_compute_random.o $(ROOT)/ve/shared/compute/bh_compute_reduce.o $(ROOT)/ve/shared/compute/bh_compute_matmul.o $(ROOT)/ve/shared/compute/bh_compute_nselect.o 
INSTALLDIR?=/opt/bohrium/

all: $(LIB)

%.o: %.c $(HEADER)
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)  -I$(ROOT)/include

%.o: %.cpp $(HEADER)
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CXX) $(OBJ) -o $@ $(LDFLAGS),$(LIB) $(CFLAGS) -I$(ROOT)/include

clean:
		rm -rf *.o *~ *.so

install: all
		cp $(LIB) $(INSTALLDIR)

