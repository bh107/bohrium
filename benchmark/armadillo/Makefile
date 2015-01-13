ROOT=../..
HEADER=$(ROOT)/include/*

BENCH_SRC  = $(wildcard src/*.cpp)
BENCHMARKS = $(subst .cpp,, $(subst src/,,$(BENCH_SRC)))

#CXX=clang
CXX=g++
EXTRAS+=
CXXFLAGS=-Wall -Wextra -pedantic -O3 -std=c++98 -Wno-variadic-macros -I$(ROOT)/include -lstdc++ -larmadillo $(EXTRAS)

all: benchmarks

benchmarks: $(BENCHMARKS)

clean:
	rm -rf *.o *~
	find bin/ -type f -name "*" -exec rm  {} \;

#
# Benchmarks
#
black_scholes: src/black_scholes.cpp $(HEADER)
	$(CXX) $< -o bin/$@ $(LCFLAGS) $(CXXFLAGS)

synth: src/synth.cpp $(HEADER)
	$(CXX) $< -o bin/$@ -L$(ROOT)/core -I$(ROOT)/include -I$(CPPB_INCLUDE) -lbh $(LCFLAGS) $(CXXFLAGS) -lstdc++ -larmadillo
