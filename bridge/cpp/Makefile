ROOT=../..
HEADER=$(ROOT)/include/*

EXAMPLE_SRC	= $(wildcard examples/*.cpp)
TEST_SRC	= $(wildcard testing/*.cpp)

TESTS		= $(subst .cpp,, $(subst testing/,,$(TEST_SRC)))
EXAMPLES	= $(subst .cpp,, $(subst examples/,,$(EXAMPLE_SRC)))
EXAMPLES=hello_world monte_carlo_pi

#CXX=clang
CXX=g++
#CXXFLAGS=-Wall -Wextra -pedantic -std=c++98 -Wno-variadic-macros
EXTRAS+=
CXXFLAGS=-Wall -Wextra -pedantic -std=c++98 -Wno-variadic-macros $(EXTRAS)

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

monte_carlo_pi: examples/monte_carlo_pi.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CXXFLAGS) -lstdc++

jacobi_stencil: examples/jacobi_stencil.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CXXFLAGS) -lstdc++

