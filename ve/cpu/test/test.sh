#!/usr/bin/env bash

WHERE=`pwd`
rm -r ~/.local/cpu
#EXTRAS="-DPROFILE" INSTALLDIR="~/.local" DEBUG="" make clean install
INSTALLDIR="~/.local" DEBUG="" make clean install
#cd ~/Desktop/benchpress/
#./press.py --suite cpu --output /tmp/ --runs 2 ../bohrium/
cd $WHERE
#python ~/Desktop/bohrium/benchmark/Python/shallow_water.py --size=3000*3000*2 --bohrium=False
#python ~/Desktop/bohrium/benchmark/Python/shallow_water.py --size=3000*3000*2 --bohrium=True
#python ~/Desktop/bohrium/benchmark/Python/shallow_water.py --size=3000*3000*2 --bohrium=True
#python ~/Desktop/bohrium/benchmark/Python/shallow_water.py --size=3000*3000*2 --bohrium=True
python ../../test/numpy/numpytest.py
#python ../../test/numpy/numpytest.py -f test_benchmarks.py
#python ../../test/numpy/numpytest.py -f test_matmul.py

#python ../../test/numpy/numpytest.py -f test_array_create.py
#python ../../test/numpy/numpytest.py -f test_primitives.py
#python ../../test/numpy/numpytest.py -f test_specials.py
#python ../../test/numpy/numpytest.py -f test_types.py
#python ../../test/numpy/numpytest.py -f test_views.py
