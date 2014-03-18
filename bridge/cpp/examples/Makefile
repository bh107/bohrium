ROOT=../../..
HEADER=$(ROOT)/include/*

EXAMPLE_SRC	= $(wildcard *.cpp)
EXAMPLES	= $(subst .cpp,, $(EXAMPLE_SRC))

CXX=g++
CXXFLAGS=-Wall -Wextra -pedantic -O2 -std=c++0x -Wno-variadic-macros $(EXTRAS)

all: $(EXAMPLES)
	@-(echo 'Done with examples.')

gen:
	./codegen/gen.py
	mv codegen/output/*.hpp bh/.

clean:
	rm -rf bin/

#
# Examples
#
hello_world: hello_world.cpp $(HEADER)
	mkdir -p bin
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I../ -lbh $(LCFLAGS) $(CXXFLAGS)


