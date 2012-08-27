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

all: mcpi

gen:
	python ./codegen/gen.py
	mv ./codegen/output/assign_const.hpp	 ./codegen/output/cphvb_cpp_traits.hpp
	cat ./codegen/output/assign_array.hpp >> ./codegen/output/cphvb_cpp_traits.hpp
	mv ./codegen/output/op_in.hpp		./codegen/output/cphvb_cppb_operators.hpp
	cat ./codegen/output/op_out.hpp >>	./codegen/output/cphvb_cppb_operators.hpp
	mv ./codegen/output/op_fun.hpp >>	./codegen/output/cphvb_cppb_functions.hpp
	mv ./codegen/output/*.hpp .

mcpi: mcpi.cpp $(HEADER) 
	$(CXX) $< -o $@ -L$(ROOT)/core -lcphvb -lrt -ldl $(LCFLAGS) $(CFLAGS) -DDEBUG

install: all
	@-(echo 'Done.')

clean:
	rm -rf *.o *~ mcpi
