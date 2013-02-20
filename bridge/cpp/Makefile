ROOT=../..
HEADER=$(ROOT)/include/*
CXX=clang
#CXX=gcc

all: gen genm hello

gen:
	./codegen/gen.py

genm:
	mv codegen/output/*.hpp .

hello: hello.cpp $(HEADER)
	$(CXX) $< -o $@ -L$(ROOT)/core -I$(ROOT)/include -lbh $(LCFLAGS) $(CFLAGS)

install: all
	@-(echo 'Done.')

clean:
	rm -rf *.o *~ hello codegen/output/*.hpp
