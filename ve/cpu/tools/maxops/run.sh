#!/usr/bin/env bash
echo "Compiling..."
gcc-4.9 -x c -std=c99 jazz.cpp -o jazz -pg -O3 -fno-inline -ftree-vectorize -march=native
echo "Running..."
/usr/bin/time ./jazz
echo "Grabbing profile data"
gprof -b ./jazz gmon.out > jazz.profile
#gprof2dot jazz.profile  > jazz.dot
#dot -T pdf jazz.dot > jazz.pdf
echo "Show it..."
cat jazz.profile
