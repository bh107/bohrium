ROOT=../..
HEADER=$(ROOT)/include/*

EXAMPLE_SRC	= $(wildcard examples/*.cpp)
TEST_SRC	= $(wildcard testing/*.cpp)

TESTS		= $(subst .cpp,, $(subst testing/,,$(TEST_SRC)))
EXAMPLES	= $(subst .cpp,, $(subst examples/,,$(EXAMPLE_SRC)))
#EXAMPLES=broadcast

#CXX=clang
CXX=g++
#CXXFLAGS=-DDEBUG
CXXFLAGS=-Wall -Wextra -pedantic -std=c++98 -Wno-variadic-macros

all: gen testing examples

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

broadcast: examples/broadcast.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CXXFLAGS)

loop: examples/loop.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CXXFLAGS)

loop_int: examples/loop_int.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CXXFLAGS)

iter: examples/iter.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CXXFLAGS)

init: examples/init.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CXXFLAGS)

printing: examples/printing.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CXXFLAGS)

#
# Testing
#
#
constructor: testing/constructor.cpp $(HEADER)
	$(CXX) $< -o bin/testing/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh -lboost_unit_test_framework $(LCFLAGS) $(CXXFLAGS)

assignment: testing/assignment.cpp $(HEADER)
	$(CXX) $< -o bin/testing/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh -lboost_unit_test_framework $(LCFLAGS) $(CXXFLAGS)

broadcasting: testing/broadcasting.cpp $(HEADER)
	$(CXX) $< -o bin/testing/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh -lboost_unit_test_framework $(LCFLAGS) $(CXXFLAGS)

compound: testing/compound.cpp $(HEADER)
	$(CXX) $< -o bin/testing/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh -lboost_unit_test_framework $(LCFLAGS) $(CXXFLAGS)

iterator: testing/iterator.cpp $(HEADER)
	$(CXX) $< -o bin/testing/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh -lboost_unit_test_framework $(LCFLAGS) $(CXXFLAGS)

operators: testing/operators.cpp $(HEADER)
	$(CXX) $< -o bin/testing/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh -lboost_unit_test_framework $(LCFLAGS) $(CXXFLAGS)


