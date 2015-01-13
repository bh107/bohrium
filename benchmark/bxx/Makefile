ROOT=../..
HEADER=$(ROOT)/include/*
CPPB_INCLUDE=$(ROOT)/bridge/cpp

BENCH_SRC  = $(wildcard src/*.cpp)
BENCHMARKS = $(subst .cpp,, $(subst src/,,$(BENCH_SRC)))
#BENCHMARKS = synth_inplace synth_branch
#BENCHMARKS = black_scholes

#CXX=clang
CXX=g++
EXTRAS+=
CXXFLAGS=-L/home/safl/.local/lib -Wall -Wextra -pedantic -g -O0 -std=c++0x -Wno-variadic-macros -I$(ROOT)/include -I$(CPPB_INCLUDE) -lbh $(LCFLAGS) -lstdc++ $(EXTRAS) 

all: benchmarks

benchmarks: $(BENCHMARKS)

clean:
	rm -rf *.o *~
	find bin/ -type f -name "*" -exec rm  {} \;

#
# Benchmarks
#
black_scholes: src/black_scholes.cpp $(HEADER)
	$(CXX) $< -o bin/$@ $(CXXFLAGS)

jacobi_stencil: src/jacobi_stencil.cpp $(HEADER)
	$(CXX) $< -o bin/$@ $(CXXFLAGS)

knn_naive: src/knn_naive.cpp $(HEADER)
	$(CXX) $< -o bin/$@ $(CXXFLAGS)

monte_carlo_pi: src/monte_carlo_pi.cpp $(HEADER)
	$(CXX) $< -o bin/$@ $(CXXFLAGS)

synth: src/synth.cpp $(HEADER)
	$(CXX) $< -o bin/$@ $(CXXFLAGS)

synth_inplace: src/synth_inplace.cpp $(HEADER)
	$(CXX) $< -o bin/$@ $(CXXFLAGS)

fusion.contraction: src/fusion.contraction.cpp $(HEADER)
	$(CXX) $< -o bin/$@ $(CXXFLAGS)

