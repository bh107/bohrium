BENCH_SRC  = $(wildcard src/*.cpp)
BENCHMARKS = $(subst .cpp,, $(subst src/,,$(BENCH_SRC)))

#CXX=clang
CXX=g++
EXTRAS+=
CXXFLAGS=-Wall -Wextra -pedantic -O2 -std=c++98 -Wno-variadic-macros $(EXTRAS)

all: benchmarks

benchmarks: $(BENCHMARKS)

clean:
	rm -rf *.o *~
	find bin/ -type f -name "*" -exec rm  {} \;

#
# Benchmarks
#
black_scholes: src/black_scholes.cpp
	$(CXX) $< -o bin/$@ $(LCFLAGS) $(CXXFLAGS) -lstdc++ -I./

