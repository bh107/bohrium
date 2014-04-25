ROOT=../..
include $(ROOT)/platform.mk
LIB=libbh_ve_cpu.$(LIB_SUFFIX)
CC?=gcc
#CXX?=g++
CXX?=clang
WARNOPTS=-Wall
EXTRAS+=
ifeq ($(PLATFORM_OS),OSX)
	EXTRAS+=-I$(ROOT)/thirdparty/ctemplate/include 
	EXTRA_LIBS+=-L$(ROOT)/thirdparty/ctemplate/osx-lib 
endif

#LCFLAGS=$(WARNOPTS) $(ARCH_OPTS) -g -O0 -fPIC -std=c++98 -I$(ROOT)/include $(EXTRAS)
LCFLAGS=$(WARNOPTS) $(ARCH_OPTS) -g -O2 -fPIC -std=c++98 -I$(ROOT)/include $(EXTRAS)
LDFLAGS=-L../../core $(ARCH_OPTS) -lbh $(EXTRA_LIBS) -lctemplate -shared -Wl,$(LD_LIB_COMMAND)
HEADER=./*.h $(ROOT)/include/*
SRC=$(ALL_CPP_FILES)
HEADER=$(ALL_H_FILES) $(ALL_HPP_FILES)
OBJ=$(SRC:.cpp=.o)
INSTALLDIR?=/opt/bohrium/

all: $(LIB)

purge:
	rm -fr $(INSTALLDIR)/cpu/

sample:
	$(BH_PYTHON) tools/sample.py 3 4 1 --be numpy
	BH_VE_CPU_JIT_DUMPSRC=1 $(BH_PYTHON) tools/sample.py 3 4 1 --be bohrium

jacobi:
	$(BH_PYTHON) ../../benchmark/Python/jacobi_stencil.py --size=10*10*1 --verbose
	BH_VE_CPU_JIT_DUMPSRC=1 $(BH_PYTHON) ../../benchmark/Python/jacobi_stencil.py --size=10*10*1 --bohrium=True --verbose

graph:
	[ -f graph-1.dot ] && dot -T svg graph-1.dot > /tmp/graph-1.svg
	[ -f /tmp/graph-1.svg ] && xdg-open /tmp/graph-1.svg
	[ -f graph-2.dot ] && dot -T svg graph-2.dot > /tmp/graph-2.svg
	[ -f /tmp/graph-2.svg ] && xdg-open /tmp/graph-2.svg
	[ -f graph-3.dot ] && dot -T svg graph-3.dot > /tmp/graph-3.svg
	[ -f /tmp/graph-3.svg ] && xdg-open /tmp/graph-3.svg

gen:
	cd autogen && ./gen.py
	cp autogen/output/block_compose.cpp .
	cp autogen/output/utils_mapping.cpp .
	cp autogen/output/specializer_cexpression.cpp .
	cp autogen/output/tac.cpp tac.h
	rm autogen/output/*.cpp

docs:
	mkdir -p doc/output
	doxygen doc/cpu_engine.doxy

test:
	BH_VE_CPU_JIT_DUMPSRC=1 dython ../../test/numpy/numpytest.py

%.o: %.cpp $(HEADER)
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS),$(LIB) $(CFLAGS) -I$(ROOT)/include

clean:
	rm -rf *.o *~ *.$(LIB_SUFFIX) kernels/* objects/* core trace-*.txt flow*.dot flow*.html graph-*.dot doc/output autogen/output/*

install: all
	cp $(LIB) $(INSTALLDIR)/
	mkdir -p $(INSTALLDIR)/cpu/kernels
	mkdir -p $(INSTALLDIR)/cpu/objects
	mkdir -p $(INSTALLDIR)/cpu/templates
	mkdir -p $(INSTALLDIR)/cpu/scripts
	cp tac.h $(INSTALLDIR)/include/
	cp templates/*.tpl $(INSTALLDIR)/cpu/templates/
	cp scripts/*.sh $(INSTALLDIR)/cpu/scripts/
	chmod 755 $(INSTALLDIR)/cpu/scripts/*
