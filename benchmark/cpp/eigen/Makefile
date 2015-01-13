include ../Makefile.common

all: benchmarks

clean:
	rm -rf *.o *~
	find bin/ -type f -name "*" -exec rm  {} \;

benchmarks: $(BENCHMARKS)

#
# Benchmarks
#
black_scholes: src/black_scholes.cpp
	$(CXX) $< -o bin/$@ $(CXXFLAGS) $(INCLUDE) $(LIBS) $(EXTRAS)

mc: src/mc.cpp
	$(CXX) $< -o bin/$@ $(CXXFLAGS) $(INCLUDE) $(LIBS) $(EXTRAS)

synth: src/synth.cpp
	$(CXX) $< -o bin/$@ $(CXXFLAGS) $(INCLUDE) $(LIBS) $(EXTRAS)

