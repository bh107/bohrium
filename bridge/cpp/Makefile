ROOT=../..
HEADER=$(ROOT)/include/*
CXX=clang
#CXX=gcc

all: gen operators hw loop

gen:
	./codegen/gen.py
	mv codegen/output/*.hpp bh/.

hw: examples/hw.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CFLAGS)

operators: examples/operators.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CFLAGS)

loop: examples/loop.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CFLAGS)

install: all
	@-(echo 'Done.')

clean:
	rm -rf *.o *~ hello codegen/output/*.hpp bin/*
