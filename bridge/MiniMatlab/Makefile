ROOT=../..
FOL=core

#CXX=clang
CXX=g++
CXXFLAGS=-Wall -std=c++0x -O2 -Wno-write-strings
FILES=$(FOL)/par.tab.cpp $(FOL)/lex.yy.c $(FOL)/AST.cpp

all:
	bison -o $(FOL)/par.tab.cpp -d $(FOL)/parser.ypp
	flex -o $(FOL)/lex.yy.c $(FOL)/lexer.l
	$(CXX) $(FILES) -o bin/MiniMatlab -L$(ROOT)/core -I$(ROOT)/include $(CXXFLAGS) -lbh 2>&1 | less -S

run:
	bin/./MiniMatlab

arithmetic:
	bin/./MiniMatlab examples/arithmetic.m

arrays:
	bin/./MiniMatlab examples/arrays.m

matrices:
	bin/./MiniMatlab examples/matrices.m

slices:
	bin/./MiniMatlab examples/slices.m

for:
	bin/./MiniMatlab examples/for_loop.m

gdb:
	gdb bin/./MiniMatlab

valgrind:
	valgrind bin/./MiniMatlab examples/test.m


