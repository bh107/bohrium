#!/usr/bin/env bash

if [ ! -z "$1" ] && [ "$1" == "reset" ]; then
    WHERE=`pwd`
    rm -r ~/.local/cpu/
    INSTALLDIR="~/.local" DEBUG="" make clean install
    cd $WHERE
fi

if [ ! -z "$1" ] && [ "$1" == "sample" ]; then
    BH_VE_CPU_JIT_DUMPSRC=1 ./sample bohrium
fi

if [ -f graph-1.dot ]; then
    dot -T svg graph-1.dot > /tmp/graph.svg
    chromium-browser /tmp/graph.svg
else
    echo "No graph to visualize."
fi

if [ ! -z "$1" ] && [ "$1" == "test" ]; then
    #BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/test_accumulate.py
    BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/numpytest.py --exclude=test_accumulate.py
fi

