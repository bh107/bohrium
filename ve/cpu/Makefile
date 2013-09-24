ROOT=../..
LIB=libbh_ve_cpu.so
CXX?=g++
#CXX?=clang
WARNOPTS=-Wall
#LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -I$(ROOT)/include -I$(ROOT)/core -I$(ROOT)/traverse
EXTRAS+=
LCFLAGS= $(WARNOPTS) -std=c++0x -g -O2 -fPIC -I$(ROOT)/include -I$(ROOT)/core -I$(ROOT)/traverse $(EXTRAS)
LDFLAGS=-L../../core -lbh -lrt -lctemplate -shared -Wl,-soname
HEADER=./*.h $(ROOT)/include/*
SRC= $(shell find -iname '*.cpp') $(shell find -iname '*.c')
OBJTMP=$(SRC:.c=.o)
OBJ=$(OBJTMP:.cpp=.o) 
OBJ=bh_ve_cpu.o $(ROOT)/ve/shared/compute/bh_compute_matmul.o $(ROOT)/ve/shared/compute/bh_compute_nselect.o 

INSTALLDIR?=/opt/bohrium/

all: $(LIB) precompiler

%.o: %.c $(HEADER)
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)  -I$(ROOT)/include

%.o: %.cpp $(HEADER)
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CXX) $(OBJ) -o $@ $(LDFLAGS),$(LIB) $(CFLAGS) -I$(ROOT)/include

precompiler: precompiler.cpp
	$(CXX)  $< -o $@ $(LIB) -I$(ROOT)/include -L../../core -lbh -lrt -ldl -ljansson -lctemplate $(LCFLAGS) $(CFLAGS)

precompile:
		/usr/bin/time ./precompiler

clean:
		rm -rf *.o *~ *.so kernels/* objects/* core precompiler
		touch kernels/empty
		touch objects/empty

install: all
		cp $(LIB) $(INSTALLDIR)/
		mkdir -p $(INSTALLDIR)/cpu/kernels
		mkdir -p $(INSTALLDIR)/cpu/objects
		mkdir -p $(INSTALLDIR)/cpu/templates
		cp templates/*.tpl $(INSTALLDIR)/cpu/templates/
		#cp objects/symbols_aaaaaa.so $(INSTALLDIR)/cpu/objects/
		#cp objects/symbols_aaaaaa.ind $(INSTALLDIR)/cpu/objects/
