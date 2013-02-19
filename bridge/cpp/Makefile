ROOT=../..
HEADER=$(ROOT)/include/*
CXX=clang

all: hello

gen:
	./codegen/gen.py

genm:
	mv codegen/output/*.hpp .

hello: hello.cpp $(HEADER)
	$(CXX) $< -o $@ -L$(ROOT)/core -I$(ROOT)/include -lbohrium $(LCFLAGS) $(CFLAGS) -DDEBUG

install: all
	@-(echo 'Done.')

clean:
	rm -rf *.o *~ hello codegen/output/*.hpp
