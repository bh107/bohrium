include ../Makefile.common
CXXFLAGS+=-fopenmp

all: benchmarks

clean:
	rm -rf *.o *~
	find bin/ -type f -name "*" -exec rm  {} \;

benchmarks: heat_equation

#
# Benchmarks
#
heat_equation: src/heat_equation.cpp
	$(CXX) $< -o bin/$@ $(CXXFLAGS) $(INCLUDE) $(LIBS) $(EXTRAS)

