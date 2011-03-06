CPHVB_BASE=../../
CUDA_BASE=/usr/local/cuda/
INCLUDE=-I$(CPHVB_BASE)include -I$(CUDA_BASE)include
LIB=libvecuda.so
CXX=g++
CC=gcc
WARNOPTS=-Wall
CFLAGS= $(WARNOPTS) -g -O2 -fPIC $(INCLUDE) -DDEBUG
CCFLAGS = $(CFLAGS) -std=gnu99
CXXFLAGS = $(CFLAGS) -std=gnu++0x
LDFLAGS=-shared -Wl,-soname,$(LIB)
CSRC=PTXopcode PTXsreg PTXtype PTXversion
CPPSRC=KernelSimple MemoryManagerSimple DataManagerSimple OffsetMapSimple \
		DeviceManager PTXconstantBuffer DeviceManagerSimple PTXconstant \
		InstructionBatchSimple PTXinstruction InstructionSchedulerSimple \
		PTXinstructionList InstructionTranslator PTXkernel Kernel \
		PTXkernelParameter KernelGeneratorSimple PTXoperand KernelParameter \
		PTXregisterBank KernelShape PTXregister KernelShapeSimple \
		PTXspecialRegister Configuration cphvb_ve_cuda 
OBJECTS=$(CPPSRC:=.oo) $(CSRC:=.o) 
INSTALLDIR=/usr/local/lib/

all: $(LIB)

%.o: %.c %.h
		$(CC) -c $< -o $@ $(CCFLAGS)

cphvb_ve_cuda.oo: cphvb_ve_cuda.cpp  cphvb_ve_cuda.h
		$(CXX) -c $< -o $@ $(CXXFLAGS)

%.oo: %.cpp %.hpp
		$(CXX) -c $< -o $@ $(CXXFLAGS)

$(LIB): $(OBJECTS)
		$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o  *.oo *~ $(LIB)

install: all
		sudo cp $(LIB) $(INSTALLDIR)

test: cudatest.c
		$(CC) $< -o $@ $(CCFLAGS) -L/usr/local/lib -L/usr/lib/nvidia -lcphvb -lvecuda -lcuda 