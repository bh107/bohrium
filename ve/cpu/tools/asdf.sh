#!/usr/bin/env bash
rm ./*.dot
rm /tmp/*.pdf

#BH_VE_CPU_BIND=1 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 python ~/bohrium/benchmark/python/heat_equation.py --size=10000*10000*10 --bohrium=False --verbose
#BH_VE_CPU_BIND=1 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 python ~/bohrium/benchmark/python/heat_equation.py --size=10000*10000*10 --bohrium=True --verbose
#BH_VE_CPU_BIND=1 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 python ~/bohrium/benchmark/python/heat_equation.py --size=10000*10000*10 --bohrium=True --verbose
BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 ./gentest.py

HEJ=`ls *.dot`
for dotfile in $HEJ
do
    dot -T pdf $dotfile > /tmp/$dotfile.pdf
    google-chrome /tmp/$dotfile.pdf &
done


