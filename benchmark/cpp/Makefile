ROOT=../..
HEADER=$(ROOT)/include/*
CPPB_INCLUDE=$(ROOT)/bridge/cpp

BENCH_SRC  = $(wildcard src/*.cpp)
#BENCHMARKS = $(subst .cpp,, $(subst src/,,$(BENCH_SRC)))
BENCHMARKS = synth_inplace 

#CXX=clang
CXX=g++
EXTRAS+=
CXXFLAGS=-Wall -Wextra -pedantic -g -O2 -std=c++0x -Wno-variadic-macros $(EXTRAS)

all: benchmarks

benchmarks: $(BENCHMARKS)

clean:
	rm -rf *.o *~
	find bin/ -type f -name "*" -exec rm  {} \;

#
# Benchmarks
#
black_scholes: src/black_scholes.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I$(CPPB_INCLUDE) -lbh $(LCFLAGS) $(CXXFLAGS) -lstdc++

synth: src/synth.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I$(CPPB_INCLUDE) -lbh $(LCFLAGS) $(CXXFLAGS) -lstdc++

synth_inplace: src/synth_inplace.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I$(CPPB_INCLUDE) -lbh $(LCFLAGS) $(CXXFLAGS) -lstdc++

jacobi_stencil: src/jacobi_stencil.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I$(CPPB_INCLUDE) -lbh $(LCFLAGS) $(CXXFLAGS) -lstdc++

knn_naive: src/knn_naive.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I$(CPPB_INCLUDE) -lbh $(LCFLAGS) $(CXXFLAGS) -lstdc++

monte_carlo_pi: src/monte_carlo_pi.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I$(CPPB_INCLUDE) -lbh $(LCFLAGS) $(CXXFLAGS) -lstdc++

