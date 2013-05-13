#!/usr/bin/env bash

BH_VE_DYNAMITE_TARGET="tcc -O2 -march=core2 -fPIC -x c -shared - -o " BH_VE_DYNAMITE_OBJECT_PATH="objects/tcc_XXXXXX" ./test/test.py
BH_VE_DYNAMITE_TARGET="gcc -O2 -march=core2 -fPIC -x c -shared - -o " BH_VE_DYNAMITE_OBJECT_PATH="objects/gcc_XXXXXX" ./test/test.py
BH_VE_DYNAMITE_TARGET="clang -O2 -march=core2 -fPIC -x c -shared - -o " BH_VE_DYNAMITE_OBJECT_PATH="objects/clan_XXXXXX" ./test/test.py
