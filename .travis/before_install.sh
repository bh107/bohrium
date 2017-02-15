#!/bin/bash
set -x
set -e

if [[ $TRAVIS_OS_NAME == "osx" ]]; then
  python --version
  ################
  #   MAC OS X   #
  ################
  # Pour some brews
  brew update
  brew install swig
  brew install homebrew/science/clblas

  # Travis already has 'boost-1.61.0_1' installed
  # brew install boost

  # Install 'python'
  case "$PYTHON_EXEC" in
    python2.7)
      # Travis already has 'python-2.7.12_1' installed
      # brew install python
      ;;
    python3.5)
      brew install -v python3
      ;;
  esac

  export DYLD_LIBRARY_PATH="/usr/lib:$HOME/.local/lib:$DYLD_LIBRARY_PATH"
  export PYTHONPATH="$HOME/.local/lib/$PYTHON_EXEC/site-packages:$PYTHONPATH"

  # Install pip-packages
  pip install --upgrade pip
  pip install numpy
  pip install Cython
  pip install cheetah
else
  #############
  #   LINUX   #
  #############
  docker pull bohrium/ubuntu:16.04
  docker build -t bohrium_release -f package/docker/bohrium.dockerfile .
fi
