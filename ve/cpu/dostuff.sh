#!/usr/bin/env bash
if [ ! -z "$1" ] && [ "$1" == "reset" ]; then
    clear && reset
    WHERE=`pwd`
    rm -r ~/.local/cpu/
    EXTRAS="-DPROFILING" INSTALLDIR="~/.local" make clean gen install
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
    python tools/move_code.py ~/.local/var/bh/kernels/ /tmp/code/sij/
fi

if [ ! -z "$1" ] && [ "$1" == "prep_fuse" ]; then
    mkdir -p /tmp/code/fuse
    rm /tmp/code/fuse/*.c
fi

if [ ! -z "$1" ] && [ "$1" == "move_fuse" ]; then
    python tools/move_code.py ~/.local/var/bh/kernels/ /tmp/code/fuse/
fi

if [ ! -z "$1" ] && [ "$1" == "sample" ]; then
    BH_VE_CPU_JIT_DUMPSRC=1 make sample
fi

if [ ! -z "$1" ] && [ "$1" == "test" ]; then
    echo "About to 'reset' and run test wo_fusion... Hit enter to continue..."
    read
    ./dostuff.sh prep_sij
    ./dostuff.sh reset
    BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../test/numpy/numpytest.py
    python tools/move_code.py ~/.local/cpu/kernels/ /tmp/code/sij/

    echo "About to 'reset' and run test w_fusion... Hit enter to continue..."
    read
    ./dostuff.sh prep_fuse
    ./dostuff.sh reset
    BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../test/numpy/numpytest.py
    python tools/move_code.py ~/.local/cpu/kernels/ /tmp/code/fuse/
fi

if [ ! -z "$1" ] && [ "$1" == "black_fused" ]; then
    ./dostuff.sh reset

    echo "** WITH Fusion ***"
    ./dostuff.sh prep_fuse
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*10 --bohrium=True
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*10 --bohrium=True
    ./dostuff.sh move_fuse

fi

if [ ! -z "$1" ] && [ "$1" == "black_sij" ]; then
    ./dostuff.sh reset

    echo "*** WITHOUT Fusion **"
    ./dostuff.sh prep_sij
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*10 --bohrium=True
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/black_scholes.py --size=5000000*10 --bohrium=True
    ./dostuff.sh move_sij

fi


if [ ! -z "$1" ] && [ "$1" == "heat_fused" ]; then
    ./dostuff.sh reset

    echo "** WITH Fusion ***"
    ./dostuff.sh prep_fuse
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/heat_equation.py --size=5000*5000*10 --bohrium=True
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/heat_equation.py --size=5000*5000*10 --bohrium=True
    ./dostuff.sh move_fuse

fi

if [ ! -z "$1" ] && [ "$1" == "heat_sij" ]; then
    echo "*** WITHOUT Fusion **"
    ./dostuff.sh prep_sij
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/heat_equation.py --size=5000*5000*10 --bohrium=True
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/heat_equation.py --size=5000*5000*10 --bohrium=True
    ./dostuff.sh move_sij

fi

if [ ! -z "$1" ] && [ "$1" == "swater" ]; then

    echo "** WITH Fusion ***"
    ./dostuff.sh reset
    ./dostuff.sh prep_fuse
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/shallow_water.py --size=4000*4000*2 --bohrium=True
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/shallow_water.py --size=4000*4000*2 --bohrium=True
    ./dostuff.sh move_fuse

    echo "That was fusion..."
    read

    echo "*** NUMPY ***"
    $BH_PYTHON ../../benchmark/Python/shallow_water.py --size=4000*4000*2 --bohrium=False
    $BH_PYTHON ../../benchmark/Python/shallow_water.py --size=4000*4000*2 --bohrium=False

    echo "That was NumPy..."
    read

    echo "*** WITHOUT Fusion **"
    ./dostuff.sh reset
    ./dostuff.sh prep_sij
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/shallow_water.py --size=4000*4000*2 --bohrium=True
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/shallow_water.py --size=4000*4000*2 --bohrium=True
    ./dostuff.sh move_sij

    echo "That was without fusion..."
    read

fi

if [ ! -z "$1" ] && [ "$1" == "synth_fused" ]; then
    #./dostuff.sh reset

    echo "** WITH Fusion ***"
    ./dostuff.sh prep_fuse
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/synth.py --size=20000000*20 --bohrium=True
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/synth.py --size=20000000*20 --bohrium=True
    ./dostuff.sh move_fuse

fi

if [ ! -z "$1" ] && [ "$1" == "synth_sij" ]; then
    #./dostuff.sh reset

    echo "*** WITHOUT Fusion **"
    ./dostuff.sh prep_sij
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/synth.py --size=20000000*20 --bohrium=True
    BH_CORE_VCACHE_SIZE=0 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 $BH_PYTHON ../../benchmark/Python/synth.py --size=20000000*20 --bohrium=True
    ./dostuff.sh move_sij

fi
