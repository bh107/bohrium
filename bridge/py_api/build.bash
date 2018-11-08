#!/usr/bin/env bash
set -e

# Script arguments:
VR_PATH=$1  # the path of the virtualenv
PYTHON_EXECUTABLE=$2 # the Python interpreter
CMAKE_BINARY_DIR=$3
CMAKE_SOURCE_DIR=$4
CMAKE_CURRENT_SOURCE_DIR=$5

echo "==================== PYTHON API ===================="

if [ -d "$VR_PATH" ]; then
  echo "Using virtualenv: $VR_PATH"
else
  echo "Create virtualenv: $VR_PATH"
  ${PYTHON_EXECUTABLE} -m virtualenv -p ${PYTHON_EXECUTABLE} ${VR_PATH}
fi

source ${VR_PATH}/bin/activate

export PY_API_LIB2INCLUDE="${CMAKE_BINARY_DIR}/*/*/libbh_*.so"
export PY_API_SRC_ROOT=${CMAKE_SOURCE_DIR}
export PY_API_BUILD_ROOT=${CMAKE_BINARY_DIR}
rm -Rf ${VR_PATH}/wheel_py_api
pip wheel -w ${VR_PATH}/wheel_py_api ${CMAKE_CURRENT_SOURCE_DIR}
WHEEL_NAME=`echo ${VR_PATH}/wheel_py_api/*`
echo "WHEEL_NAME: \"$WHEEL_NAME\""
pip install -I ${WHEEL_NAME}
echo -n "$WHEEL_NAME" > ${VR_PATH}/wheel_py_api/package_name.txt

deactivate
echo  "===================================================="