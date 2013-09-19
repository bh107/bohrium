ROOT=../../..
HEADER=$(ROOT)/include/*

CXX=clang
#CXX=g++
EXTRAS+=
#CXXFLAGS=-Wall -Wextra -pedantic -O2 -std=c++98 -Wno-variadic-macros $(EXTRAS)
CXXFLAGS=-Wall -Wextra -pedantic -O2 -std=c++0x -Wno-variadic-macros $(EXTRAS)

crash: examples/crash.cpp $(HEADER)
	$(CXX) $< -o $@ -L$(ROOT)/core -I$(ROOT)/include -I./ -lbh $(LCFLAGS) $(CXXFLAGS)

run: 
	./crash
