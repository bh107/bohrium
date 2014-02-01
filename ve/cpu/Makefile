ROOT=../..
include $(ROOT)/platform.mk
LIB=libbh_ve_cpu.$(LIB_SUFFIX)
CXX?=g++
#CXX?=clang
WARNOPTS=-Wall
EXTRAS+=
ifeq ($(PLATFORM_OS),OSX)
	EXTRAS+=-I$(ROOT)/thirdparty/ctemplate/include 
	EXTRA_LIBS+=-L$(ROOT)/thirdparty/ctemplate/osx-lib 
endif

LCFLAGS=$(WARNOPTS) $(ARCH_OPTS) -std=c++0x -g -O2 -fPIC -I$(ROOT)/include -I$(ROOT)/core $(EXTRAS)
LDFLAGS=-L../../core $(ARCH_OPTS) -lbh $(EXTRA_LIBS) -lctemplate -shared -Wl,$(LD_LIB_COMMAND)
HEADER=./*.h $(ROOT)/include/*
SRC= $(ALL_CPP_FILES) $(ALL_C_FILES)
OBJTMP=$(SRC:.c=.o)
OBJ=$(OBJTMP:.cpp=.o) 
OBJ=bh_ve_cpu.o 

INSTALLDIR?=/opt/bohrium/

all: $(LIB)

purge:
	rm -fr $(INSTALLDIR)/cpu/

sample:
	python misc/sample.py 3 4 2 1 --be numpy
	BH_VE_CPU_JIT_DUMPSRC=1 python misc/sample.py 3 4 2 1 --be bohrium

jacobi:
	python ../../benchmark/Python/jacobi_stencil.py --size=10*10*1 --verbose
	BH_VE_CPU_JIT_DUMPSRC=1 python ../../benchmark/Python/jacobi_stencil.py --size=10*10*1 --bohrium=True --verbose

graph:
	[ -f graph-1.dot ] && dot -T svg graph-1.dot > /tmp/graph-1.svg
	[ -f /tmp/graph-1.svg ] && xdg-open /tmp/graph-1.svg
	[ -f graph-2.dot ] && dot -T svg graph-2.dot > /tmp/graph-2.svg
	[ -f /tmp/graph-2.svg ] && xdg-open /tmp/graph-2.svg
	[ -f graph-3.dot ] && dot -T svg graph-3.dot > /tmp/graph-3.svg
	[ -f /tmp/graph-3.svg ] && xdg-open /tmp/graph-3.svg

test:
	BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/numpytest.py -f test_accumulate.py

%.o: %.c $(HEADER)
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)  -I$(ROOT)/include

%.o: %.cpp $(HEADER)
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS),$(LIB) $(CFLAGS) -I$(ROOT)/include

clean:
	rm -rf *.o *~ *.$(LIB_SUFFIX) kernels/* objects/* core trace-*.txt graph-*.dot

install: all
	cp $(LIB) $(INSTALLDIR)/
	mkdir -p $(INSTALLDIR)/cpu/kernels
	mkdir -p $(INSTALLDIR)/cpu/objects
	mkdir -p $(INSTALLDIR)/cpu/templates
	cp templates/*.tpl $(INSTALLDIR)/cpu/templates/
