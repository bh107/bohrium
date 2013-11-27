BH_BASE=../../
BH_INCLUDE=$(BH_BASE)include/
INCLUDE=-I$(BH_BASE)include
LIB=libbh_ve_gpu.so
CXX=g++
CC=gcc
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O3 -fPIC $(INCLUDE) $(CFLAGS)  #-DDEBUG #-DSTATS
CCFLAGS = $(LCFLAGS) -std=gnu99
CXXFLAGS = $(LCFLAGS) -std=gnu++0x -D__CL_ENABLE_EXCEPTIONS

LDFLAGS = -L../../core -L/usr/lib/nvidia-current -lOpenCL -lbh -shared -Wl,-soname,$(LIB)
CSRC = OCLtype
CPPSRC = ResourceManager bh_ve_gpu InstructionBatch Buffer KernelParameter BaseArray Scalar \
		InstructionScheduler Kernel GenerateSourceCode Reduce UserFunctionRandom \
		UserFunctionMatmul UserFunctionLu UserFunctionFft StringHasher
 
OBJECTS=$(CPPSRC:=.oo) $(CSRC:=.o) $(ROOT)/ve/shared/compute/bh_compute_reduce.o 
INSTALLDIR?=/opt/bohrium/

all: $(LIB)

%.o: %.c %.h Makefile
		$(CC) -c $< -o $@ $(CCFLAGS)

bh_ve_gpu.oo: bh_ve_gpu.cpp bh_ve_gpu.h Makefile
		$(CXX) -c $< -o $@ $(CXXFLAGS)

%.oo: %.cpp %.hpp Makefile
		$(CXX) -c $< -o $@ $(CXXFLAGS)

$(LIB): $(OBJECTS)
		$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o  *.oo *~ *.gch *.orig $(LIB)

install: all
		cp $(LIB) $(INSTALLDIR)
		cp -r ocl_source $(INSTALLDIR)/lib/

