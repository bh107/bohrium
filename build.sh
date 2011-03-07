#!/bin/bash
set -e #Force bash script to exit on the first command error.

#Bash script for building the whole cphVB .

#Change current directory to the root of cphVB and this script is located.
cd `dirname "$0"`

PYTHON="/usr/bin/python"
CUDA=0
CLUSTER=0
BRIDGE=""

while getopts "M:E:C:" opt; do
  case $opt in
    M)
      HIGH=`echo $OPTARG  | tr '[a-z]' '[A-Z]'`
      LOW=`echo $OPTARG  | tr '[A-Z]' '[a-z]'`
      export CFLAGS="-D$HIGH $CFLAGS"
      CLUSTER=1
      ;;
    E)
      HIGH=`echo $OPTARG  | tr '[a-z]' '[A-Z]'`
      LOW=`echo $OPTARG  | tr '[A-Z]' '[a-z]'`
      export CFLAGS="-D$HIGH $CFLAGS"
      CUDA=1
      ;;
    C)
      PYTHON="$OPTARG"
      BRIDGE="$BRIDGE -C$OPTARG"
      ;;
    \?)
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done


#Change current directory to where numpy and this script is located.
cd `dirname "$0"`
if [ "$CLUSTER" -eq "1" ]
  then
    echo "***Building VEM-CLUSTER***"
    cd vem/cluster
    make clean all
    cd ../..
    BRIDGE="$BRIDGE -MCLUSTER"
fi
echo "***Building VEM-NODE***"
cd vem/node
make clean all
cd ../../
if [ "$CUDA" -eq "1" ]
  then
    echo "***Building VE-CUDA***"
    cd ve/cuda
    make all
    cd ../../
    BRIDGE="$BRIDGE -ECUDA"
fi
echo "***Building VE-SIMPLE***"
cd ve/simple
make all
cd ../../
echo "***Building CORE***"
cd core
make all
cd ..
echo "***Building NUMPY_BRIDGE***"
bridge/numpy/build.sh -C$PYTHON $BRIDGE
