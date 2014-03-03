BENCH_SRC  = $(wildcard src/*.chpl)
BENCHMARKS = $(subst .chpl,, $(subst src/,,$(BENCH_SRC)))

CHPL=chpl
CHPL_FLGS=--savec ccode/

all: $(BENCHMARKS)

%:
	$(CHPL) -o bin/$@ $(CHPL_FLGS) src/$@.chpl

clean:
	rm -rf *.o *~
	find bin/ -type f -name "*" -exec rm  {} \;

