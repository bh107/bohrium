#!/usr/bin/env bash

INSTALLDIR="~/.local" DEBUG="" make clean install
# tcc seem to be having problems with stdarg.h
#BH_VE_DYNAMITE_TARGET="tcc -DTCC_TARGET_X86_64 -O2 -march=core2 -fPIC -x c -shared - -o " BH_VE_DYNAMITE_OBJECT_PATH="objects/tcc_XXXXXX" ./test/test.py
#BH_VE_DYNAMITE_TARGET="gcc -std=c99 -O2 -march=native -fPIC -x c -shared - -o " BH_VE_DYNAMITE_OBJECT_PATH="objects/gcc_" ./test/test.py
#BH_VE_DYNAMITE_TARGET="clang -std=c99 -O2 -march=native -fPIC -x c -shared - -o " BH_VE_DYNAMITE_OBJECT_PATH="objects/clang_" ./test/test.py

./test/test.py

#python ../../test/numpy/numpytest.py -f test_array_create.py
#python ../../test/numpy/numpytest.py -f test_benchmarks.py
#python ../../test/numpy/numpytest.py -f test_matmul.py
#python ../../test/numpy/numpytest.py -f test_primitives.py
#python ../../test/numpy/numpytest.py -f test_specials.py
#python ../../test/numpy/numpytest.py -f test_types.py
#python ../../test/numpy/numpytest.py -f test_views.py
