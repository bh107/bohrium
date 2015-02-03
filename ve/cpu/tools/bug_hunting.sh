#!/usr/bin/env bash

make purge_krn install

echo "NUMPY"
BH_VE_CPU_JIT_DUMPSRC=1 BH_VE_CPU_JIT_FUSION=0 python ~/bohrium/benchmark/Python/synth_strided.py --size=100000000*10 --bohrium=False
BH_VE_CPU_JIT_DUMPSRC=1 BH_VE_CPU_JIT_FUSION=0 python ~/bohrium/benchmark/Python/synth_strided.py --size=100000000*10 --bohrium=False

echo "BH-FUSION-MC"
BH_VE_CPU_JIT_DUMPSRC=1 BH_VE_CPU_JIT_FUSION=1 python ~/bohrium/benchmark/Python/synth_strided.py --size=100000000*10 --bohrium=True
BH_VE_CPU_JIT_DUMPSRC=1 BH_VE_CPU_JIT_FUSION=1 python ~/bohrium/benchmark/Python/synth_strided.py --size=100000000*10 --bohrium=True
BH_VE_CPU_JIT_DUMPSRC=1 BH_VE_CPU_JIT_FUSION=1 python ~/bohrium/benchmark/Python/synth_strided.py --size=100000000*10 --bohrium=True

echo "BH-FUSION-SC"
OMP_NUM_THREADS=1 BH_VE_CPU_JIT_DUMPSRC=1 BH_VE_CPU_JIT_FUSION=1 python ~/bohrium/benchmark/Python/synth_strided.py --size=100000000*10 --bohrium=True
OMP_NUM_THREADS=1 BH_VE_CPU_JIT_DUMPSRC=1 BH_VE_CPU_JIT_FUSION=1 python ~/bohrium/benchmark/Python/synth_strided.py --size=100000000*10 --bohrium=True
OMP_NUM_THREADS=1 BH_VE_CPU_JIT_DUMPSRC=1 BH_VE_CPU_JIT_FUSION=1 python ~/bohrium/benchmark/Python/synth_strided.py --size=100000000*10 --bohrium=True

echo "BH-SIJ-MC"
BH_VE_CPU_JIT_DUMPSRC=1 BH_VE_CPU_JIT_FUSION=0 python ~/bohrium/benchmark/Python/synth_strided.py --size=100000000*10 --bohrium=True
BH_VE_CPU_JIT_DUMPSRC=1 BH_VE_CPU_JIT_FUSION=0 python ~/bohrium/benchmark/Python/synth_strided.py --size=100000000*10 --bohrium=True
BH_VE_CPU_JIT_DUMPSRC=1 BH_VE_CPU_JIT_FUSION=0 python ~/bohrium/benchmark/Python/synth_strided.py --size=100000000*10 --bohrium=True

echo "BH-SIJ-SC"
OMP_NUM_THREADS=1 BH_VE_CPU_JIT_DUMPSRC=1 BH_VE_CPU_JIT_FUSION=0 python ~/bohrium/benchmark/Python/synth_strided.py --size=100000000*10 --bohrium=True
OMP_NUM_THREADS=1 BH_VE_CPU_JIT_DUMPSRC=1 BH_VE_CPU_JIT_FUSION=0 python ~/bohrium/benchmark/Python/synth_strided.py --size=100000000*10 --bohrium=True
OMP_NUM_THREADS=1 BH_VE_CPU_JIT_DUMPSRC=1 BH_VE_CPU_JIT_FUSION=0 python ~/bohrium/benchmark/Python/synth_strided.py --size=100000000*10 --bohrium=True


