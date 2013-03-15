ROOT=../..
HEADER=$(ROOT)/include/*
SOURCES=$(wildcard examples/*.cpp)
OBJECTS=$(subst .cpp,.o, $(SOURCES))
EXAMPLES=$(subst .cpp,, $(subst examples/,,$(SOURCES)))
#EXAMPLES=broadcast
#CXX=clang
CXX=g++
#CXXFLAGS=-Wall
CXXFLAGS=-DDEBUG

all: $(EXAMPLES)

gen:
	./codegen/gen.py
	mv codegen/output/*.hpp bh/.

print:
	echo $(EXAMPLES)

#$(EXAMPLES): $(SOURCES)
#	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CXXFLAGS)

#$(SOURCES): $(OBJECTS)
#	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CXXFLAGS)

hw: examples/hw.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CXXFLAGS)

broadcast: examples/broadcast.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CXXFLAGS)


compound: examples/compound.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CXXFLAGS)

constructors: examples/constructors.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CXXFLAGS)

operators: examples/operators.cpp $(HEADER)
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

install: gen $(EXAMPLES)
	@-(echo 'Done.')

clean:
	rm -rf *.o *~ hello codegen/output/*.hpp bin/*
