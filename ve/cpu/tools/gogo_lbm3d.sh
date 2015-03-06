#!/usr/bin/env bash
HOSTNAME=`hostname`

echo "NumPy"
OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 python ~/bohrium/benchmark/python/lbm_3d.py --size=150*150*150*10 --bohrium=False > $HOSTNAME.np.txt
echo "SIJ"
OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 python ~/bohrium/benchmark/python/lbm_3d.py --size=150*150*150*10 --bohrium=True > $HOSTNAME.sij1.txt
OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=0 BH_VE_CPU_JIT_DUMPSRC=1 python ~/bohrium/benchmark/python/lbm_3d.py --size=150*150*150*10 --bohrium=True > $HOSTNAME.sij2.txt
echo "Fusion"
OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 python ~/bohrium/benchmark/python/lbm_3d.py --size=150*150*150*10 --bohrium=True > $HOSTNAME.fused1.txt
OMP_NUM_THREADS=1 BH_VE_CPU_JIT_FUSION=1 BH_VE_CPU_JIT_DUMPSRC=1 python ~/bohrium/benchmark/python/lbm_3d.py --size=150*150*150*10 --bohrium=True > $HOSTNAME.fused2.txt

