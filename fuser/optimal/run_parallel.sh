#!/usr/bin/env bash

SIZE=$1
BENCHMARK=$2
FUSE_MODEL=$3
CONFIGFILE=$4
EXEC_CMD=$5

function spawn_job () {

    echo "Spawn job ${BENCHMARK}-${FUSE_MODEL} [$1]"
    echo "
    #SBATCH -J ${BENCHMARK}-${FUSE_MODEL}
    #SBATCH -o ${BENCHMARK}-${FUSE_MODEL}-%j.out
    #SBATCH --share
    #SBATCH -p low
    export BH_VE_CPU_BIND=0
    export OMP_NUM_THREADS=1
    export BH_CONFIG=${CONFIGFILE}
    export BH_FUSE_MODEL=${FUSE_MODEL}
    export BH_FUSER_OPTIMAL_ORDER=regular
    export BH_FUSER_OPTIMAL_PRELOAD=${1}
    ${EXEC_CMD}
    " > tmp.slurm_job
    #sbatch tmp.slurm_job
    cat tmp.slurm_job
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
