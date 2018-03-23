#!/bin/bash

# The script pulls the bohrium repos and use create_wheel.py to create a PIP package
# Use this script on a MAC OSX platform with homebrew and python installed.
#
# Command line arguments:
#   1) Set the first argument to the name of the branch to pull from
#   2) Set the second argument to the python version e.g. 2.7 or 3.6
#   3) Set the third argument to "testing" if you want to test the wheel package
#   4) Set the fourth argument to "deploy" if you want to upload the wheel package to PIP,
#      in which case you need to set the envs TWINE_USERNAME and TWINE_PASSWORD

set -e
set -x
unset PYTHONPATH
export BH_OPENMP_PROF=true
export BH_OPENMP_VOLATILE=true
export BH_OPENCL_PROF=true
export BH_OPENCL_VOLATILE=true
export HOMEBREW_NO_AUTO_UPDATE=1

if [ "$#" -ne "4" ];
    then echo "illegal number of parameters -- e.g. master 2.7 testing nodeploy"
fi

# Making sure that the python version is installed
python$2 --version
which python$2

# Install dependencies
python$2 -m pip install --user numpy cython twine scipy
brew install cmake || true
brew install boost --with-icu4c || true
brew install libsigsegv || true
brew install clblas || true
brew install opencv || true

# Download source into `~/bh`
git clone https://github.com/bh107/bohrium.git --branch $1
mv bohrium ~/bh

# Create a script `~/bh/build.sh` that build and install Bohrium with the given python version
echo "mkdir -p ~/bh/b\$1 && cd ~/bh/b\$1 && cmake .. -DCMAKE_BUILD_TYPE=Release -DEXT_VISUALIZER=OFF -DVEM_PROXY=OFF -DPYTHON_EXECUTABLE=/usr/local/bin/python\$1 -DCMAKE_INSTALL_PREFIX=~/bh/i\$1 -DFORCE_CONFIG_PATH=~/bh/i\$1 && make install" > ~/bh/build.sh

# Create a script `/bh/wheel.sh` that build a Python wheel of the given python version
echo "cd ~/bh/b\$1 && python\$1 ~/bh/package/pip/create_wheel.py --npbackend-dir ~/bh/i\$1/lib/python\$1/site-packages/bohrium/ --bh-install-prefix ~/bh/i\$1 --lib-dir-name lib --config ~/bh/i\$1/config.ini bdist_wheel \\" > ~/bh/wheel.sh

# Include GNU libsigsegv
echo "-L/usr/local/lib/libsigsegv.2.dylib \\" >> ~/bh/wheel.sh

# Include boost
echo "-L/usr/local/opt/boost/lib/libboost_serialization-mt.dylib \\" >> ~/bh/wheel.sh
echo "-L/usr/local/opt/boost/lib/libboost_filesystem-mt.dylib \\" >> ~/bh/wheel.sh
echo "-L/usr/local/opt/boost/lib/libboost_system-mt.dylib \\" >> ~/bh/wheel.sh
echo "-L/usr/local/opt/boost/lib/libboost_regex-mt.dylib \\" >> ~/bh/wheel.sh

# Include clBLAS
echo "-L/usr/local/lib/libclBLAS.dylib " >> ~/bh/wheel.sh

# Build Bohrium and the wheel package
bash ~/bh/build.sh $2
bash ~/bh/wheel.sh $2
echo "Build package:" && ls ~/bh/b$2/dist/*

# Testing of the wheel package
if [ "$3" = "testing" ]; then
    python$2 -m pip install ~/bh/b$2/dist/*
    python$2 -m bohrium --info

    # We have to skip some tests because of time constraints on travis-ci.org
    set +x
    TESTS=""
    for t in `ls ~/bh/test/python/tests/test_*.py`; do
        if ! [[ $t =~ (mask|reorganization|summations) ]]; then
            TESTS="$TESTS $t"
        fi
    done
    set -x
    BH_STACK=openmp python$2 ~/bh/test/python/run.py $TESTS
else
    echo 'Notice, if you want to run test set third argument to "testing"'
fi

# Deploy, remember to define TWINE_USERNAME and TWINE_PASSWORD
if [ "$4" = "deploy" ]; then
    python$2 -m twine upload ~/bh/b*/dist/* || true
else
    echo 'Notice, if you want to run test set fourth argument to "deploy"'
fi

