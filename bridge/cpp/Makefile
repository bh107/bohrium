ROOT=../..
HEADER=$(ROOT)/include/*

all: hello

hello: hello.cpp $(HEADER)
	$(CXX) $< -o $@ -L$(ROOT)/core -lbohrium $(LCFLAGS) $(CFLAGS) -DDEBUG

install: all
	@-(echo 'Done.')

clean:
	rm -rf *.o *~ hello
