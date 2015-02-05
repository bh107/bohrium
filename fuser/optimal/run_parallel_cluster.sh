#!/usr/bin/env bash

#run example on the whole cluster:
#~/bohrium/fuser/optimal/run_parallel_cluster.sh 3 nbody same_shape_range_random /home/madsbk/fusecache/all/bh_config.ini "python /home/madsbk/bohrium/benchmark/Python/nbody.py --size=100*1 --bohrium=True"

SIZE=$1
BENCHMARK=$2
FUSE_MODEL=$3
CONFIGFILE=$4
EXEC_CMD=$5

JOBFILE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/run_parallel_one_node.slurm"

function spawn_job () {
    echo "sbatch job ${BENCHMARK}-${FUSE_MODEL} [$1]"
    sbatch ${JOBFILE} $((SIZE + 5)) ${BENCHMARK} ${FUSE_MODEL} ${SIZE} "${EXEC_CMD}" "${1}"

}

function permut () {
    if [ ${#1} -lt $SIZE ]; then
        permut  "${1}1"
        permut  "${1}0"
    else
        spawn_job ${1}
    fi
}
permut ""
