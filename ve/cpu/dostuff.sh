#!/usr/bin/env bash

if [ ! -z "$1" ] && [ "$1" == "reset" ]; then
    WHERE=`pwd`
    rm -r ~/.local/cpu/
    INSTALLDIR="~/.local" make clean gen install
    cd $WHERE
fi

if [ ! -z "$1" ] && [ "$1" == "deset" ]; then
    WHERE=`pwd`
    rm -r ~/.local/cpu/
    INSTALLDIR="~/.local" EXTRAS="-DDEBUGGING" make clean gen install
    cd $WHERE
fi

if [ ! -z "$1" ] && [ "$1" == "sample" ]; then
    BH_VE_CPU_JIT_DUMPSRC=1 make sample
fi

if [ -f graph-1.dot ]; then
    dot -T svg graph-1.dot > /tmp/graph.svg
    chromium-browser /tmp/graph.svg
else
    echo "No graph to visualize."
fi

if [ ! -z "$1" ] && [ "$1" == "test" ]; then
    BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/numpytest.py
    #BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/numpytest.py --exclude test_matmul.py --exclude test_ndstencil.py
    #BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/numpytest.py -f test_matmul.py
    #BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/numpytest.py -f test_accumulate.py
    #BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/numpytest.py -f test_benchmarks.py
    #BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/numpytest.py -f test_array_create.py
    #BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/numpytest.py -f test_ndstencil.py
    #BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/numpytest.py -f test_primitives.py
    #BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/numpytest.py -f test_specials.py
    #BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/numpytest.py -f test_sor.py
    #BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/numpytest.py -f test_types.py
    #BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/numpytest.py -f test_views.py
    #BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/numpytest.py -f test_reduce.py
fi

