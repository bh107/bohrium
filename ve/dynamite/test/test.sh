#!/usr/bin/env bash

#INSTALLDIR="~/.local" DEBUG="" make clean install
# tcc seem to be having problems with stdarg.h
#BH_VE_DYNAMITE_TARGET="tcc -DTCC_TARGET_X86_64 -O2 -march=core2 -fPIC -x c -shared - -o " BH_VE_DYNAMITE_OBJECT_PATH="objects/tcc_XXXXXX" ./test/test.py
#BH_VE_DYNAMITE_TARGET="gcc -std=c99 -O2 -march=native -fPIC -x c -shared - -o " BH_VE_DYNAMITE_OBJECT_PATH="objects/gcc_" ./test/test.py
#BH_VE_DYNAMITE_TARGET="clang -std=c99 -O2 -march=native -fPIC -x c -shared - -o " BH_VE_DYNAMITE_OBJECT_PATH="objects/clang_" ./test/test.py

#cp test/config.ini.naive ~/.bohrium/config.ini
#./test/test.py
#
#cp test/config.ini.simple ~/.bohrium/config.ini
#./test/test.py
WHERE=`pwd`
rm -r ~/.local/dynamite
#EXTRAS="-DPROFILE" INSTALLDIR="~/.local" DEBUG="" make clean install
INSTALLDIR="~/.local" DEBUG="" make clean install
#BH_CORE_VCACHE_SIZE=1 BH_VE_DYNAMITE_DOFUSE="1" ~/Desktop/bohrium/benchmark/cpp/bin/monte_carlo_pi --size=10000000*20 --verbose
#BH_CORE_VCACHE_SIZE=1 BH_VE_DYNAMITE_DOFUSE="1" ~/Desktop/bohrium/benchmark/cpp/bin/monte_carlo_pi --size=10000000*20 --verbose
cd ~/Desktop/benchpress/
./press.py --suite dynamite --output /tmp/ --runs 2 ../bohrium/
cd $WHERE



#python ~/Desktop/bohrium/benchmark/Python/mc.py    --size=1000000*100 --bohrium=False
#python ~/Desktop/bohrium/benchmark/Python/mc.py --size=1000000*100 --bohrium=True
#python ~/Desktop/bohrium/benchmark/Python/mc.py --size=1000000*100 --bohrium=True
#python ~/Desktop/bohrium/benchmark/Python/mc.py --size=10000000*10 --bohrium=True
#python ../../test/numpy/numpytest.py

#OMP_NUM_THREADS=1 python ~/Desktop/bohrium/benchmark/Python/mc.py --size=10000000*10 --bohrium=True
#OMP_NUM_THREADS=1 python ~/Desktop/bohrium/benchmark/Python/mc.py --size=10000000*10 --bohrium=True
#OMP_NUM_THREADS=2 python ~/Desktop/bohrium/benchmark/Python/mc.py --size=10000000*10 --bohrium=True
#OMP_NUM_THREADS=4 python ~/Desktop/bohrium/benchmark/Python/mc.py --size=10000000*10 --bohrium=True
#python ../../test/numpy/numpytest.py
#python ../../test/numpy/numpytest.py -f test_array_create.py
#python ../../test/numpy/numpytest.py -f test_benchmarks.py
#python ../../test/numpy/numpytest.py -f test_matmul.py
#python ../../test/numpy/numpytest.py -f test_primitives.py
#python ../../test/numpy/numpytest.py -f test_specials.py
#python ../../test/numpy/numpytest.py -f test_types.py
#python ../../test/numpy/numpytest.py -f test_views.py
