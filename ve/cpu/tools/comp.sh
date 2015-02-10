#!/usr/bin/env bash
#g++ cc_difftype.cpp -std=c++11 -I /home/safl/.local/include/ -I /home/safl/bohrium/include/ -L /home/safl/.local/lib/ -lbh -o cdt
#g++ cc_sametype.cpp -std=c++11 -I /home/safl/.local/include/ -I /home/safl/bohrium/include/ -L /home/safl/.local/lib/ -lbh -o cst
#g++ aliasing.cpp -std=c++11 -I /home/safl/.local/include/ -I /home/safl/bohrium/include/ -L /home/safl/.local/lib/ -lbh -o aliasing
#g++ dptr.cpp -std=c++11 -I /home/safl/.local/include/ -I /home/safl/bohrium/include/ -L /home/safl/.local/lib/ -lbh -o dptr

BH_VE_CPU_JIT_DUMPSRC=1 BH_VE_CPU_JIT_FUSION=0 python ~/bohrium/benchmark/Python/synth_strided.py --size=100000000*10 --bohrium=True



