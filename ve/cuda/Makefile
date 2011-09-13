CPHVB_BASE=../../
CPHVB_INCLUDE=$(CPHVB_BASE)include/
CUDA_BASE=/usr/local/cuda/
INCLUDE=-I$(CPHVB_BASE)include -I$(CUDA_BASE)include
LIB=libcphvb_ve_cuda.so
CXX=g++
CC=gcc
NVCC=nvcc
NVCCFLAGS=-ptx -arch=sm_12
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC $(INCLUDE) $(CFLAGS)
CCFLAGS = $(LCFLAGS) -std=gnu99
CXXFLAGS = $(LCFLAGS) -std=gnu++0x
LDFLAGS=-shared -Wl,-soname,$(LIB)
CSRC=PTXopcode PTXsreg PTXtype PTXversion CUDAerrorCode
CPPSRC=KernelSimple MemoryManagerSimple DataManagerSimple OffsetMapSimple \
		DeviceManager DeviceManagerSimple PTXconstant \
		InstructionBatchSimple PTXinstruction InstructionSchedulerSimple \
		InstructionTranslator PTXkernel Kernel KernelGenerated \
		KernelPredefined ReduceTB \
		PTXkernelParameter KernelGeneratorSimple PTXoperand KernelParameter \
		PTXregisterBank KernelShape PTXregister KernelShapeSimple \
		PTXspecialRegister Configuration cphVBarray cphVBinstruction \
		RandomNumberGeneratorHybridTaus \
		cphvb_ve_cuda

OBJECTS=$(CPPSRC:=.oo) $(CSRC:=.o)
INSTALLDIR=/usr/local/lib/
KERNELDIR=/usr/local/cphvb/

all: $(LIB) htrand.ptx tbreduce.ptx

%.o: %.c %.h
		$(CC) -c $< -o $@ $(CCFLAGS)

cphvb_ve_cuda.oo: cphvb_ve_cuda.cpp  $(CPHVB_INCLUDE)cphvb_ve_cuda.h
		$(CXX) -c $< -o $@ $(CXXFLAGS)

%.oo: %.cpp %.hpp
		$(CXX) -c $< -o $@ $(CXXFLAGS)

%.ptx:	%.cu
		$(NVCC) $(NVCCFLAGS) $<

$(LIB): $(OBJECTS)
		$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o  *.oo *~ *.gch *.orig *.ptx $(LIB)

install: all
		sudo cp $(LIB) $(INSTALLDIR)
		sudo mkdir -p $(KERNELDIR)
		sudo cp *.ptx $(KERNELDIR)

