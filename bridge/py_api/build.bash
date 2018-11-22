#!/usr/bin/env bash
set -e

# Script arguments:
VR_PATH=$1  # the path of the virtualenv
PY_EXE=$2 # the Python interpreter to use when building the wheel
PY_WHEEL=$3
CMAKE_BINARY_DIR=$4
CMAKE_SOURCE_DIR=$5
CMAKE_CURRENT_SOURCE_DIR=$6
CMAKE_INSTALL_PREFIX=$7

echo "==================== PYTHON API ===================="

if [[ -d "$VR_PATH" ]]; then
  echo "Using virtualenv: $VR_PATH"
else
  echo "Create virtualenv: $VR_PATH"
  ${PY_EXE} -m virtualenv -p ${VR_PATH}
fi

source ${VR_PATH}/bin/activate

# bohrium_api depend on gcc7 on MacOSX
if [[ $OSTYPE == darwin* ]]; then
    pip install gcc7
fi

export PY_API_WHEEL=${PY_WHEEL}
export PY_API_SRC_ROOT=${CMAKE_SOURCE_DIR}
export PY_API_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
export PY_API_BUILD_ROOT=${CMAKE_BINARY_DIR}
rm -Rf ${VR_PATH}/wheel_py_api
pip wheel --no-deps --verbose -w ${VR_PATH}/wheel_py_api ${CMAKE_CURRENT_SOURCE_DIR}
WHEEL_NAME=`echo ${VR_PATH}/wheel_py_api/bohrium_api-*.whl`
echo "WHEEL_NAME: \"$WHEEL_NAME\""
pip install --no-deps -I ${WHEEL_NAME}
echo -n "$WHEEL_NAME" > ${VR_PATH}/wheel_py_api/package_name.txt

deactivate
echo  "===================================================="