#!/bin/bash
set -e #Force bash script to exit on the first command error.

#Bash script for building the whole cphVB .

#Change current directory to the root of cphVB and this script is located.
cd `dirname "$0"`

PYTHON=${PYTHON-"/usr/bin/python"}
CUDA=0
CLUSTER=0
BRIDGE=""
preBUILD=""
posBUILD=""
INSTALL=0

while getopts "M:E:C:D:" opt; do
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
    D)
      export CFLAGS="-D$OPTARG $CFLAGS"
      BRIDGE="$BRIDGE -D$OPTARG"
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

# Decrements the argument pointer so it points to next argument.
# $1 now references the first non-option item supplied on the command-line
# if one exists.
shift $(($OPTIND - 1))

for arg in $@
  do
    if [ "$arg" = "clean" ]
      then
        posBUILD="$posBUILD clean"
    fi
    if [ "$arg" = "install" ]
      then
        posBUILD="$posBUILD install"
        INSTALL=1
    fi
    if [ "$arg" = "all" ]
      then
        posBUILD="$posBUILD all"
    fi
done

#Change current directory to where numpy and this script is located.
echo "***Building CORE***"
cd core
$preBUILD make $posBUILD
cd ..

if [ "$CLUSTER" -eq "1" ]
  then
    echo "***Building VEM-CLUSTER***"
    cd vem/cluster
    make $posBUILD
    cd ../..
    BRIDGE="$BRIDGE -MCLUSTER"
fi
echo "***Building VEM-NODE***"
cd vem/node
$preBUILD make $posBUILD
cd ../../
if [ "$CUDA" -eq "1" ]
  then
    echo "***Building VE-CUDA***"
    cd ve/cuda
    $preBUILD make $posBUILD
    cd ../../
    BRIDGE="$BRIDGE -ECUDA"
fi
echo "***Building VE-CUDA***"
cd ve/cuda
$preBUILD make $posBUILD
cd ../../
echo "***Building VE-SIMPLE***"
cd ve/simple
$preBUILD make $posBUILD
cd ../../
echo "***Building VE-SCORE***"
cd ve/score
$preBUILD make $posBUILD
cd ../../
echo "***Building VE-MCORE***"
cd ve/mcore
$preBUILD make $posBUILD
cd ../../
echo "***Building INIPARSER***"
cd iniparser
$preBUILD make $posBUILD
cd ..
echo "***Building NUMPY_BRIDGE***"
cd bridge/numpy
./build.sh -C$PYTHON $BRIDGE $posBUILD
cd ../../

if [ "$INSTALL" -eq "1" ]
  then
    sudo ldconfig
fi
