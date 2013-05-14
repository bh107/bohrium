#!/usr/bin/env bash

INSTALLDIR="~/.local" DEBUG="" make clean install
# tcc seem to be having problems with stdarg.h
#BH_VE_DYNAMITE_TARGET="tcc -DTCC_TARGET_X86_64 -O2 -march=core2 -fPIC -x c -shared - -o " BH_VE_DYNAMITE_OBJECT_PATH="objects/tcc_XXXXXX" ./test/test.py
BH_VE_DYNAMITE_TARGET="gcc -O2 -march=native -fPIC -x c -shared - -o " BH_VE_DYNAMITE_OBJECT_PATH="objects/gcc_" ./test/test.py
BH_VE_DYNAMITE_TARGET="clang -O2 -march=native -fPIC -x c -shared - -o " BH_VE_DYNAMITE_OBJECT_PATH="objects/clang_" ./test/test.py
