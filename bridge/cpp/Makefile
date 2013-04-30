ROOT=../..
HEADER=$(ROOT)/include/*

EXAMPLE_SRC	= $(wildcard examples/*.cpp)
EXAMPLES	= $(subst .cpp,, $(subst examples/,,$(EXAMPLE_SRC)))

CXX=clang
#CXX=g++
EXTRAS+=
CXXFLAGS=-Wall -Wextra -pedantic -O2 -std=c++98 -Wno-variadic-macros $(EXTRAS)

all: gen examples

examples: $(EXAMPLES)
testing: $(TESTS)

print:
	echo $(TESTS)

install: gen
	@-(echo 'Done.')

clean:
	rm -rf *.o *~ codegen/output/*.hpp
	find bin/ -type f -name "*" -exec rm  {} \;

gen:
	./codegen/gen.py
	mv codegen/output/*.hpp bh/.

#
# Examples
#
hello_world: examples/hello_world.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CXXFLAGS)

