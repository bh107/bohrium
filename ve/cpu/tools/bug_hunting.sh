#!/usr/bin/env bash

make purge_ko install
BH_VE_CPU_JIT_DUMPSRC=1 BH_VE_CPU_JIT_FUSION=1 python ~/bohrium/test/numpy/numpytest.py --exclude-test=test_largedim --exclude-test=test_different_inputs
BH_VE_CPU_JIT_DUMPSRC=1 BH_VE_CPU_JIT_FUSION=0 python ~/bohrium/test/numpy/numpytest.py

# This fails when contracting as hinted by bh_ir_kernel temps
#python ~/bohrium/test/numpy/numpytest.py --test=test_largedim --test=test_shallow_water --test=test_different_inputs
