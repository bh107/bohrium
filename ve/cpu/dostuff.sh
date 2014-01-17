#!/usr/bin/env bash

WHERE=`pwd`
rm -r ~/.local/cpu/

INSTALLDIR="~/.local" DEBUG="" make clean install
cd $WHERE
#BH_VE_CPU_JIT_PRELOAD=0 BH_VE_CPU_JIT_ENABLED=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_OPTIMIZE=0 BH_VE_CPU_JIT_DUMPSRC=1 ../../bridge/cpp/examples/bin/hello_world
#BH_VE_CPU_JIT_PRELOAD=1 BH_VE_CPU_JIT_ENABLED=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_OPTIMIZE=0 BH_VE_CPU_JIT_DUMPSRC=1 python -c 'import bohrium as np; a = np.random.random([3,3,3], dtype=np.float64, bohrium=True); print a' 
#BH_VE_CPU_JIT_PRELOAD=1 BH_VE_CPU_JIT_ENABLED=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_OPTIMIZE=0 BH_VE_CPU_JIT_DUMPSRC=1 python -c 'import bohrium as np; a = np.random.random([27], dtype=np.float64, bohrium=True); print a' 
#BH_VE_CPU_JIT_PRELOAD=0 BH_VE_CPU_JIT_ENABLED=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_OPTIMIZE=0 BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/numpytest.py -f test_reduce.py
#../../bridge/cpp/bin/hello_world

BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/test_accumulate.py
BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/numpytest.py
#python ../../test/numpy/numpytest.py -f test_benchmarks.py
#python ../../test/numpy/numpytest.py -f test_matmul.py
#python ../../test/numpy/numpytest.py -f test_array_create.py
#python ../../test/numpy/numpytest.py -f test_primitives.py
#python ../../test/numpy/numpytest.py -f test_specials.py
#python ../../test/numpy/numpytest.py -f test_types.py
#python ../../test/numpy/numpytest.py -f test_views.py
