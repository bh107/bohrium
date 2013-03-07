ROOT=../..
HEADER=$(ROOT)/include/*
SOURCES=$(wildcard examples/*.cpp)
OBJECTS=$(subst .cpp,.o, $(SOURCES))
EXAMPLES=$(subst .cpp,, $(subst examples/,,$(SOURCES)))
#CXX=clang
CXX=g++

all: $(EXAMPLES)

gen:
	./codegen/gen.py
	mv codegen/output/*.hpp bh/.

print:
	echo $(EXAMPLES)

#$(EXAMPLES): $(SOURCES)
#	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CFLAGS)

#$(SOURCES): $(OBJECTS)
#	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CFLAGS)

hw: examples/hw.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CFLAGS)

operators: examples/operators.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CFLAGS)

loop: examples/loop.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CFLAGS)

iter: examples/iter.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CFLAGS)

init: examples/init.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CFLAGS)

printing: examples/printing.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CFLAGS)

install: gen $(EXAMPLES)
	@-(echo 'Done.')

clean:
	rm -rf *.o *~ hello codegen/output/*.hpp bin/*
