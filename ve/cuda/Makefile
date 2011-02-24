CPHVB_BASE=../../
CUDA_BASE=/usr/local/cuda/
INCLUDE=-I$(CPHVB_BASE)include -I$(CUDA_BASE)include
LIB=libvecuda.so
CXX=g++
CC=gcc
WARNOPTS=-Wall
CFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=gnu++0x $(INCLUDE)
LDFLAGS=-shared -Wl,-soname,$(LIB)

MODULES=DataManager DeviceManager InstructionBatch InstructionScheduler \
		MemoryManager
IMPL=$(MODULES:=Simple)
TYPES=PTXconstant PTXregister PTXoperand
CONTAINERS=PTXconstantBuffer PTXregisterBank
OBJECTS=$(IMPL:=.o) $(TYPES:=.o) $(CONTAINERS:=.o) 

all: $(LIB)

%Simple.o: %Simple.cpp %Simple.hpp %.hpp
		$(CXX) -c $< -o $@ $(CFLAGS)

%.o: %.cpp %.hpp
		$(CXX) -c $< -o $@ $(CFLAGS)

$(LIB): $(OBJECTS)
		$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)
