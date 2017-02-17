#!/bin/bash
set -x
set -e

if [[ $TRAVIS_OS_NAME == "osx" ]]; then
  ################
  #   MAC OS X   #
  ################
  # Fetch 'benchpress'
  mkdir -p $HOME/benchpress
  wget -nv https://github.com/bh107/benchpress/archive/master.zip -O $HOME/benchpress/master.zip
  pushd $HOME/benchpress
  unzip -q master.zip
  popd

  export PATH="$HOME/benchpress/benchpress-master/bin:$PATH"
  export DYLD_LIBRARY_PATH="/usr/lib:$HOME/.local/lib:$DYLD_LIBRARY_PATH"
  export PYTHONPATH="$HOME/benchpress/benchpress-master/module:$PYTHONPATH"
  export PYTHONPATH="$HOME/.local/lib/$PYTHON_EXEC/site-packages:$PYTHONPATH"

  # Build 'bohrium'
  cmake . -DCMAKE_BUILD_TYPE=Debug -DUSE_WERROR=ON -DEXT_VISUALIZER=OFF
  make install
else
  #############
  #   LINUX   #
  #############
  echo "Nothing special for 'linux'."
  echo "We just run everything in docker."
fi
