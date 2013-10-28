ROOT=../..
LIB=libbhc.so
CXX?=g++
WARNOPTS=-Wall -Wno-c++11-extensions
LCFLAGS= $(WARNOPTS) $(ARCHOPTS) -g -O2 -fPIC -std=c++98 -I$(ROOT)/include -I../cpp/bh
LDFLAGS= -shared -ldl -Wl,-soname,$(LIB)

SRC=bh_c_implementation_basics.cpp bh_c_implementation.cpp
OBJ=$(SRC:.cpp=.oo) $(ROOT)/core/libbh.so

all: $(LIB)

gen: codegen/templates/*.ctpl codegen/*.json codegen/*.py
	./codegen/gen.py
	mv codegen/output/*.h include/.
	mv codegen/output/*.hpp .
	mv codegen/output/*.cpp .

%.oo: %.cpp $(ROOT)/include/* ../cpp/bh *.hpp *.cpp include/*.h gen
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ) gen
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

clean:
	rm -rf *.o *~ $(LIB)
	rm -rf *.o *~ codegen/output/*
	find lib/ -type f -name "*" -exec rm  {} \;

install: all
	cp $(LIB) $(INSTALLDIR)




