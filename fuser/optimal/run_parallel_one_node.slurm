#!/usr/bin/env bash

SIZE=$1
BENCHMARK=$2
FUSE_MODEL=$3
CONFIGFILE=$4
EXEC_CMD=$5
PERMUT=$6

export BH_VE_CPU_BIND=0
export OMP_NUM_THREADS=1
export BH_CONFIG=${CONFIGFILE}
export BH_FUSE_MODEL=${FUSE_MODEL}
export BH_FUSER_OPTIMAL_ORDER=regular

COUNT=0
function spawn_job () {
    echo "[$COUNT] spawn job ${BENCHMARK}-${FUSE_MODEL} [$1]"
    BH_FUSER_OPTIMAL_PRELOAD=${1} taskset -c ${COUNT} ${EXEC_CMD} &> ${BENCHMARK}-${FUSE_MODEL}-${SLURM_JOB_ID}-${1}.out &
    COUNT=$((COUNT+1))
}

function permut () {
    if [ ${#1} -lt $SIZE ]; then
        permut  "${1}1"
        permut  "${1}0"
    else
        spawn_job ${1}
    fi
}
permut "${PERMUT}"
sleep infinity
