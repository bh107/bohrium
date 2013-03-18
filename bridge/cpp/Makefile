ROOT=../..
HEADER=$(ROOT)/include/*

EXAMPLE_SRC	= $(wildcard examples/*.cpp)
TEST_SRC	= $(wildcard testing/*.cpp)

TESTS		= $(subst .cpp,, $(subst testing/,,$(TEST_SRC)))
EXAMPLES	= $(subst .cpp,, $(subst examples/,,$(EXAMPLE_SRC)))
#EXAMPLES=broadcast

CXX=clang
#CXX=g++
#CXXFLAGS=-DDEBUG
#CXXFLAGS=-Wall -Wextra -pedantic -std=c++98 -Wno-variadic-macros
CXXFLAGS=-Wall -Wextra -pedantic -std=c++0x -Wno-variadic-macros -Wno-newline-eof

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

slicing: examples/slicing.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CXXFLAGS) -lstdc++


