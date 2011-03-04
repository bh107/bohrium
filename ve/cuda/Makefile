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
CFILES=PTXopcode PTXsreg PTXtype PTXversion  cphvb_ve_cuda
CPPFILES=KernelSimple MemoryManagerSimple DataManagerSimple OffsetMapSimple \
		DeviceManager PTXconstantBuffer DeviceManagerSimple PTXconstant \
		InstructionBatchSimple PTXinstruction InstructionSchedulerSimple \
		PTXinstructionList InstructionTranslator PTXkernel Kernel \
		PTXkernelParameter KernelGeneratorSimple PTXoperand KernelParameter \
		PTXregisterBank KernelShape PTXregister KernelShapeSimple \
		PTXspecialRegister
OBJECTS=$(CPPFILES:=.oo) $(CFILES:=.o) 

all: $(LIB)

%.o: %.c %.h
		$(CC) -c $< -o $@ $(CCFLAGS)

%.oo: %.cpp %.hpp
		$(CXX) -c $< -o $@ $(CXXFLAGS)

$(LIB): $(OBJECTS)
		$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o  *.oo *~ $(LIB)
