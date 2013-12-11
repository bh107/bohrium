ROOT=../..
include $(ROOT)/platform.mk
LIB=libbhc.$(LIB_SUFFIX)
CXX?=g++
WARNOPTS=-Wall -Wno-c++11-extensions
LCFLAGS= $(WARNOPTS) $(ARCH_OPTS) -g -O2 -fPIC -std=c++0x -I$(ROOT)/include -I../cpp/bh
LDFLAGS= $(ARCH_OPTS) -shared -ldl -Wl,$(LD_LIB_COMMAND),$(LIB)

SRC=$(ALL_CPP_FILES)
OBJ=$(SRC:.cpp=.oo) $(ROOT)/core/libbh.$(LIB_SUFFIX)

all: gen $(LIB)

gen: codegen/templates/*.ctpl codegen/*.json codegen/*.py
	./codegen/gen.py
	mkdir -p include
	mkdir -p output
	mv codegen/output/*.h include/.
	mv codegen/output/*.hpp .
	mv codegen/output/*.cpp .

%.oo: %.cpp $(ROOT)/include/* ../cpp/bh *.hpp *.cpp include/*.h gen
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ) gen
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

clean:
	rm -rf *.o *.oo *~ $(LIB)
	rm -rf include output
	rm -rf codegen/output/*

install: all
	cp $(LIB) $(INSTALLDIR)




