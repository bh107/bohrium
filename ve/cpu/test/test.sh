#!/usr/bin/env bash

WHERE=`pwd`
rm -r ~/.local/cpu
INSTALLDIR="~/.local" DEBUG="" make clean install
cd $WHERE
#python ./test/3d_reduction.py]
#python ./test/crap.py
#python ~/Desktop/bohrium/benchmark/Python/shallow_water.py --size=3000*3000*2 --bohrium=False
#python ~/Desktop/bohrium/benchmark/Python/shallow_water.py --size=3000*3000*2 --bohrium=True
#python ~/Desktop/bohrium/benchmark/Python/shallow_water.py --size=3000*3000*2 --bohrium=True
#python ~/Desktop/bohrium/benchmark/Python/shallow_water.py --size=3000*3000*2 --bohrium=True
python ../../test/numpy/numpytest.py
#python ../../test/numpy/numpytest.py -f test_benchmarks.py
#python ../../test/numpy/numpytest.py -f test_matmul.py
#python ../../test/numpy/numpytest.py -f test_array_create.py
#python ../../test/numpy/numpytest.py -f test_reduce.py
#python ../../test/numpy/numpytest.py -f test_primitives.py
#python ../../test/numpy/numpytest.py -f test_specials.py
#python ../../test/numpy/numpytest.py -f test_types.py
#python ../../test/numpy/numpytest.py -f test_views.py
