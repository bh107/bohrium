ROOT=../..
include $(ROOT)/platform.mk
LIB=libbhc.$(LIB_SUFFIX)
CXX?=g++
WARNOPTS=-Wall -Wno-c++11-extensions
CFLAGS= $(WARNOPTS) $(ARCH_OPTS) -g -O2 -fPIC -std=c++0x -I$(ROOT)/include -I../cpp/bh -I./codegen/output/
LDFLAGS= $(ARCH_OPTS) -shared -ldl -L$(ROOT)/core -lbh -Wl,$(LD_LIB_COMMAND),$(LIB)

SRC=codegen/output/bh_c_implementation_basics.cpp codegen/output/bh_c_implementation.cpp
OBJ=$(SRC:.cpp=.oo)

all: $(LIB)

gen: codegen/templates/*.ctpl codegen/*.json codegen/*.py
	./codegen/gen.py

%.oo: %.cpp $(ROOT)/include/* ../cpp/bh/*.hpp
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): gen $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

clean:
	rm -rf *.o *.oo *~ $(LIB)
	rm -rf include output
	rm -rf codegen/output/*

install: all
	cp $(LIB) $(INSTALLDIR)




