#!/usr/bin/env bash
if [ ! -z "$1" ] && [ "$1" == "reset" ]; then
    clear && reset
    WHERE=`pwd`
    rm -r ~/.local/cpu/
    INSTALLDIR="~/.local" make clean gen install
    cd $WHERE
fi

if [ ! -z "$1" ] && [ "$1" == "deset" ]; then
    clear && reset
    WHERE=`pwd`
    rm -r ~/.local/cpu/
    INSTALLDIR="~/.local" EXTRAS="-DDEBUGGING" make clean gen install
    cd $WHERE
fi

if [ ! -z "$1" ] && [ "$1" == "prep_sij" ]; then
    mkdir -p /tmp/code/sij
    rm /tmp/code/sij/*.c
fi

if [ ! -z "$1" ] && [ "$1" == "move_sij" ]; then
    python tools/move_code.py ~/.local/cpu/kernels/ /tmp/code/sij/
fi

if [ ! -z "$1" ] && [ "$1" == "prep_fuse" ]; then
    mkdir -p /tmp/code/fuse
    rm /tmp/code/fuse/*.c
fi

if [ ! -z "$1" ] && [ "$1" == "move_fuse" ]; then
    python tools/move_code.py ~/.local/cpu/kernels/ /tmp/code/fuse/
fi

if [ ! -z "$1" ] && [ "$1" == "sample" ]; then
    BH_VE_CPU_JIT_DUMPSRC=1 make sample
fi

if [ ! -z "$1" ] && [ "$1" == "test" ]; then
    echo "About to 'reset' and run test w_fusion... Hit enter to continue..."
    read
    ./dostuff.sh prep_fuse
    ./dostuff.sh reset
    BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../test/numpy/numpytest.py --exclude=test_ndstencil.py
    #BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../test/numpy/numpytest.py --exclude=test_benchmarks.py
    #BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../test/numpy/numpytest.py
    python tools/move_code.py ~/.local/cpu/kernels/ /tmp/code/fuse/

    echo "About to 'reset' and run test wo_fusion... Hit enter to continue..."
    read
    ./dostuff.sh prep_sij
    ./dostuff.sh reset
    BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../test/numpy/numpytest.py
    python tools/move_code.py ~/.local/cpu/kernels/ /tmp/code/sij/
fi

if [ ! -z "$1" ] && [ "$1" == "black" ]; then
    ./dostuff.sh reset

    echo "** WITH Fusion ***"
    ./dostuff.sh prep_fuse
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*10 --bohrium=True
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*10 --bohrium=True
    $BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*10 --bohrium=True
    ./dostuff.sh move_fuse

    echo "*** NUMPY ***"
    $BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*10 --bohrium=False
    #$BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*2 --bohrium=False
    #$BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*2 --bohrium=False

    echo "*** WITHOUT Fusion **"
    ./dostuff.sh prep_sij
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*10 --bohrium=True
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*10 --bohrium=True
    BH_VE_CPU_JIT_FUSION=0 $BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*10 --bohrium=True
    ./dostuff.sh move_sij

fi

if [ ! -z "$1" ] && [ "$1" == "jacobi" ]; then
    ./dostuff.sh reset

    echo "** WITH Fusion ***"
    ./dostuff.sh prep_fuse
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/jacobi_stencil.py --size=10000*10000*5 --bohrium=True
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/jacobi_stencil.py --size=10000*10000*5 --bohrium=True
    $BH_PYTHON ../../benchmark/Python/jacobi_stencil.py --size=10000*10000*5 --bohrium=True
    ./dostuff.sh move_fuse

    echo "*** NUMPY ***"
    $BH_PYTHON ../../benchmark/Python/jacobi_stencil.py --size=10000*10000*5 --bohrium=False
    #$BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*2 --bohrium=False
    #$BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*2 --bohrium=False

    echo "*** WITHOUT Fusion **"
    ./dostuff.sh prep_sij
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/jacobi_stencil.py --size=10000*10000*5 --bohrium=True
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/jacobi_stencil.py --size=10000*10000*5 --bohrium=True
    BH_VE_CPU_JIT_FUSION=0 $BH_PYTHON ../../benchmark/Python/jacobi_stencil.py --size=10000*10000*5 --bohrium=True
    ./dostuff.sh move_sij

fi

if [ ! -z "$1" ] && [ "$1" == "swater" ]; then
    ./dostuff.sh reset

    echo "** WITH Fusion ***"
    ./dostuff.sh prep_fuse
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/shallow_water.py --size=4000*4000*2 --bohrium=True
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/shallow_water.py --size=4000*4000*2 --bohrium=True
    $BH_PYTHON ../../benchmark/Python/shallow_water.py --size=4000*4000*2 --bohrium=True
    ./dostuff.sh move_fuse

    echo "*** NUMPY ***"
    $BH_PYTHON ../../benchmark/Python/shallow_water.py --size=4000*4000*2 --bohrium=False
    $BH_PYTHON ../../benchmark/Python/shallow_water.py --size=4000*4000*2 --bohrium=False

    echo "*** WITHOUT Fusion **"
    ./dostuff.sh prep_sij
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/shallow_water.py --size=4000*4000*2 --bohrium=True
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/shallow_water.py --size=4000*4000*2 --bohrium=True
    BH_VE_CPU_JIT_FUSION=0 $BH_PYTHON ../../benchmark/Python/shallow_water.py --size=4000*4000*2 --bohrium=True
    ./dostuff.sh move_sij

fi
