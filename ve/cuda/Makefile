CPHVB_BASE=../../
CUDA_BASE=/usr/local/cuda/
INCLUDE=-I$(CPHVB_BASE)include -I$(CUDA_BASE)include
LIB=libvecuda.so
CXX=g++
CC=gcc
WARNOPTS=-Wall
CFLAGS= $(WARNOPTS) -g -O2 -fPIC $(INCLUDE)
CCFLAGS = $(CFLAGS) -std=gnu99
CXXFLAGS = $(CFLAGS) -std=gnu++0x
LDFLAGS=-shared -Wl,-soname,$(LIB)

MODULES=DataManager DeviceManager InstructionBatch InstructionScheduler \
		MemoryManager
IMPL=$(MODULES:=Simple)
TYPES=PTXconstant PTXregister PTXoperand
CONTAINERS=PTXconstantBuffer PTXregisterBank
OBJECTS=$(IMPL:=.oo) $(TYPES:=.oo) $(CONTAINERS:=.oo) $(CFILES:=.o) 

all: $(LIB)

%Simple.o: %Simple.cpp %Simple.hpp %.hpp
		$(CXX) -c $< -o $@ $(CXXFLAGS)

%.o: %.c %.h
		$(CC) -c $< -o $@ $(CCFLAGS)

%.oo: %.cpp %.hpp
		$(CXX) -c $< -o $@ $(CXXFLAGS)

$(LIB): $(OBJECTS)
		$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o  *.oo *~ $(LIB)
