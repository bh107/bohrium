#!/usr/bin/env bash
rm ./*.dot
rm /tmp/*.pdf

BH_VE_CPU_BIND=1 OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 python ~/bohrium/benchmark/python/mc.py --size=60000000*3 --bohrium=True --verbose

HEJ=`ls *.dot`
for dotfile in $HEJ
do
    dot -T pdf $dotfile > /tmp/$dotfile.pdf
    google-chrome /tmp/$dotfile.pdf &
done


