CPHVB_BASE=../../
CPHVB_INCLUDE=$(CPHVB_BASE)include/
INCLUDE=-I$(CPHVB_BASE)include
LIB=libcphvb_ve_gpu.so
CXX=g++
CC=gcc
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O3 -fPIC $(INCLUDE) $(CFLAGS)  #-DDEBUG #-DSTATS
CCFLAGS = $(LCFLAGS) -std=gnu99
CXXFLAGS = $(LCFLAGS) -std=gnu++0x -D__CL_ENABLE_EXCEPTIONS

LDFLAGS = -L../../core -L/usr/lib/nvidia-current -lOpenCL -lcphvb -shared -Wl,-soname,$(LIB)
CSRC = OCLtype
CPPSRC = ResourceManager cphvb_ve_gpu InstructionBatch Buffer KernelParameter BaseArray Scalar \
		InstructionScheduler Kernel GenerateSourceCode UserFunctionRandom UserFunctionReduce \
		UserFunctionMatmul UserFunctionLu UserFunctionFft StringHasher
 
OBJECTS=$(CPPSRC:=.oo) $(CSRC:=.o)
INSTALLDIR?=/opt/cphvb/

all: $(LIB)

%.o: %.c %.h Makefile
		$(CC) -c $< -o $@ $(CCFLAGS)

cphvb_ve_gpu.oo: cphvb_ve_gpu.cpp cphvb_ve_gpu.h Makefile
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

