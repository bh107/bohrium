#!/usr/bin/env bash
set -e

# Script arguments:
VR_PATH=$1  # the path of the virtualenv
PYTHON_EXECUTABLE=$2 # the Python interpreter
CMAKE_BINARY_DIR=$3
CMAKE_CURRENT_SOURCE_DIR=$4

export LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bridge/c:${CMAKE_BINARY_DIR}/bridge/cxx:${CMAKE_BINARY_DIR}/core:${LD_LIBRARY_PATH}
export DYLD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${DYLD_LIBRARY_PATH}

echo "===================== NPBACKEND ======================"

source ${VR_PATH}/bin/activate
pip install numpy cython
rm -Rf ${VR_PATH}/wheel_npbackend
pip wheel --no-deps --verbose -w ${VR_PATH}/wheel_npbackend ${CMAKE_CURRENT_SOURCE_DIR}
WHEEL_NAME=`echo ${VR_PATH}/wheel_npbackend/bohrium-*.whl`
echo "WHEEL_NAME: \"$WHEEL_NAME\""
pip install --no-deps -I ${WHEEL_NAME}
echo -n "$WHEEL_NAME" > ${VR_PATH}/wheel_npbackend/package_name.txt

deactivate
echo  "===================================================="