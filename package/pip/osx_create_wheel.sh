#!/bin/bash

# Use this script on a MAC OSX platform with homebrew installed.
# The script pulls the bohrium repos and use create_wheel.py to create a PIP package
#
# Command line arguments:
#   1) Set the first argument to the name of the branch to pull from
#   2) Set the second argument to "deploy" if you want to upload the wheel packages to PIP,
#      in which case you need to set the envs TWINE_USERNAME and TWINE_PASSWORD

set -e
set -x
unset PYTHONPATH

# Install non-python dependencies
brew install cmake || true
brew install boost --with-icu4c || true
brew install libsigsegv || true
#brew install opencv
#brew install clblas

# Download source into `~/bh`
git clone https://github.com/madsbk/bohrium.git --branch $1
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
echo "-L/usr/local/opt/boost/lib/libboost_regex-mt.dylib " >> ~/bh/wheel.sh

# Build Bohrium with python2.7
brew unlink python || true
brew install python@2
brew link python@2 || true
/usr/local/bin/python2.7 -m pip install numpy cython
bash ~/bh/build.sh 2.7
bash ~/bh/wheel.sh 2.7

# Build Bohrium with python3.6
brew unlink python || true
brew install python@3
brew link python@3 || true
/usr/local/bin/python3.6 -m pip install numpy cython
bash ~/bh/build.sh 3.6
bash ~/bh/wheel.sh 3.6

echo "Build packages:"
ls ~/bh/b*/dist/*

# Deploy, remember to define TWINE_USERNAME and TWINE_PASSWORD
if [ "$2" = "deploy" ]; then
    python3.6 -m pip install --user twine
    python3.6 -m twine upload ~/bh/b*/dist/*
else
    echo "Notice, if you want to upload the wheel packages use: osx_create_wheel.sh deploy"
fi

