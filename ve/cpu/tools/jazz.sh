BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 OMP_NUM_THREADS=1 ./gentest.py 1
dot -T pdf dag-1.dot > /tmp/hej.pdf
#evince /tmp/hej.pdf

