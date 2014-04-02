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

    echo "About to 'reset' and run test wo_fusion... Hit enter to continue..."
    read
    clear && reset
    mkdir -p /tmp/code/sij
    rm /tmp/code/sij/*.c
    ./dostuff.sh reset
    BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../test/numpy/numpytest.py
    python tools/move_code.py ~/.local/cpu/kernels/ /tmp/code/sij/

    echo "About to 'reset' and run test w_fusion... Hit enter to continue..."
    read
    clear && reset
    mkdir -p /tmp/code/fuse
    rm /tmp/code/fuse/*.c
    ./dostuff.sh reset
    BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../test/numpy/numpytest.py
    python tools/move_code.py ~/.local/cpu/kernels/ /tmp/code/fuse/
fi

if [ ! -z "$1" ] && [ "$1" == "fusion" ]; then
    echo "About to 'reset' and run test fusion... Hit enter to continue..."
    read
    clear && reset
    mkdir -p /tmp/code/fuse
    rm /tmp/code/fuse/*.c
    ./dostuff.sh reset
    #BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 python ../../test/numpy/numpytest.py
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 python ../../benchmark/Python/black_scholes.py --size=5000000*2 --bohrium=True
    python tools/move_code.py ~/.local/cpu/kernels/ /tmp/code/fuse/
fi

if [ ! -z "$1" ] && [ "$1" == "black" ]; then
    ./dostuff.sh reset
    clear && reset

    echo "*** NUMPY ***"
    $BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*2 --bohrium=False
    $BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*2 --bohrium=False
    $BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*2 --bohrium=False

    echo "*** WITHOUT Fusion **"
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON../../benchmark/Python/black_scholes.py --size=5000000*2 --bohrium=True
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*2 --bohrium=True
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*2 --bohrium=True

    echo "** WITH Fusion ***"
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*2 --bohrium=True
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*2 --bohrium=True
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*2 --bohrium=True

fi

