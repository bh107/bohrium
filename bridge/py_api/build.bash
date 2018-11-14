#!/usr/bin/env bash
set -e

# Script arguments:
VR_PATH=$1  # the path of the virtualenv
PYTHON_EXECUTABLE=$2 # the principle Python interpreter
PY_EXE=$3 # the Python interpreter to use when building the wheel
PY_WHEEL=$4
CMAKE_BINARY_DIR=$5
CMAKE_SOURCE_DIR=$6
CMAKE_CURRENT_SOURCE_DIR=$7
CMAKE_INSTALL_PREFIX=$8

echo "==================== PYTHON API ===================="

if [[ -d "$VR_PATH" ]]; then
  echo "Using virtualenv: $VR_PATH"
else
  echo "Create virtualenv: $VR_PATH"
  ${PYTHON_EXECUTABLE} -m virtualenv -p ${PY_EXE} ${VR_PATH}
fi

source ${VR_PATH}/bin/activate

export PY_API_WHEEL=${PY_WHEEL}
export PY_API_SRC_ROOT=${CMAKE_SOURCE_DIR}
export PY_API_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
export PY_API_BUILD_ROOT=${CMAKE_BINARY_DIR}
rm -Rf ${VR_PATH}/wheel_py_api
pip wheel --verbose -w ${VR_PATH}/wheel_py_api ${CMAKE_CURRENT_SOURCE_DIR}
WHEEL_NAME=`echo ${VR_PATH}/wheel_py_api/bohrium_api-*.whl`
echo "WHEEL_NAME: \"$WHEEL_NAME\""
pip install -I ${WHEEL_NAME}
echo -n "$WHEEL_NAME" > ${VR_PATH}/wheel_py_api/package_name.txt

deactivate
echo  "===================================================="