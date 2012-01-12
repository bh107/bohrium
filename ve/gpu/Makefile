CPHVB_BASE=../../
CPHVB_INCLUDE=$(CPHVB_BASE)include/
INCLUDE=-I$(CPHVB_BASE)include
LIB=libcphvb_ve_gpu.so
CXX=g++-4.6
CC=gcc
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC $(INCLUDE) $(CFLAGS)
CCFLAGS = $(LCFLAGS) -std=gnu99
CXXFLAGS = $(LCFLAGS) -std=gnu++0x -D__CL_ENABLE_EXCEPTIONS

LDFLAGS = -shared -Wl,-soname,$(LIB)
CSRC = oclType
CPPSRC = ResourceManager cphvb_ve_gpu InstructionBatch ArrayOperand BaseArray View InstructionScheduler

OBJECTS=$(CPPSRC:=.oo) $(CSRC:=.o)
INSTALLDIR=/usr/local/lib/
KERNELDIR=/usr/local/cphvb/

all: $(LIB)

%.o: %.c %.h
		$(CC) -c $< -o $@ $(CCFLAGS)

cphvb_ve_gpu.oo: cphvb_ve_gpu.cpp cphvb_ve_gpu.h
		$(CXX) -c $< -o $@ $(CXXFLAGS)

%.oo: %.cpp %.hpp
		$(CXX) -c $< -o $@ $(CXXFLAGS)

$(LIB): $(OBJECTS)
		$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o  *.oo *~ *.gch *.orig $(LIB)

install: all
		sudo cp $(LIB) $(INSTALLDIR)
		sudo mkdir -p $(KERNELDIR)
		sudo cp *.ptx $(KERNELDIR)

