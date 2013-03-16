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

run:
	@echo 'Running Tests'
	@for TEST in $(TESTS); do bin/testing/$$TEST --log_level=error --report_level=no; done

#
# Examples
#
hello_world: examples/hello_world.cpp $(HEADER)
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

loop_scope: testing/loop_scope.cpp $(HEADER)
	$(CXX) $< -o bin/testing/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh -lboost_unit_test_framework $(LCFLAGS) $(CXXFLAGS)

