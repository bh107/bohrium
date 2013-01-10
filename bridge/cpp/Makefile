ROOT=../..
CXX?=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -std=c++98 -fPIC -I$(ROOT)/include
LDFLAGS=-L$(ROOT)/core -lcphvb -lrt -shared -Wl,-soname
HEADER=$(ROOT)/include/*
SRC= $(shell find -iname '*.cpp') $(shell find -iname '*.c')
OBJTMP=$(SRC:.c=.o)
OBJ=$(OBJTMP:.cpp=.o)
INSTALLDIR?=/opt/cphvb/

all: examples

gen:
	python ./codegen/gen.py
	mv ./codegen/output/*.hpp .

examples: examples/hello.cpp $(HEADER) 
	$(CXX) $< -o $@ -L$(ROOT)/core -lcphvb -lrt -ldl $(LCFLAGS) $(CFLAGS) -DDEBUG

install: all
	@-(echo 'Done.')

clean:
	rm -rf *.o *~ hello mcpi codegen/output/*.cpp codegen/output/*.hpp
