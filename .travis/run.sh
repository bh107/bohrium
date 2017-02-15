#!/bin/bash
set -x
set -e

if [[ $TRAVIS_OS_NAME == "osx" ]]; then
  ################
  #   MAC OS X   #
  ################
  export PATH="$HOME/benchpress/benchpress-master/bin:$PATH"
  export DYLD_LIBRARY_PATH="/usr/lib:$HOME/.local/lib:$DYLD_LIBRARY_PATH"
  export PYTHONPATH="$HOME/benchpress/benchpress-master/module:$PYTHONPATH"
  export PYTHONPATH="$HOME/.local/lib/$PYTHON_EXEC/site-packages:$PYTHONPATH"

  # The 'python' binaries are not called 'python2.7' and 'python3.5' from Brew
  case "$PYTHON_EXEC" in
    python2.7)
      export PEXEC="python"
      ;;
    python3.5)
      export PEXEC="python3"
      ;;
  esac

  $PEXEC $TRAVIS_BUILD_DIR/test/python/run.py $TRAVIS_BUILD_DIR/test/python/tests/test_*.py
  $PEXEC $TRAVIS_BUILD_DIR/test/python/numpytest.py --file $TRAVIS_BUILD_DIR/test/python/test_benchmarks.py
else
  #############
  #   LINUX   #
  #############
  docker run -t -e BH_STACK -e BH_OPENMP_PROF -e BH_OPENCL_PROF -e PYTHON_EXEC -e BH_OPENMP_VOLATILE -e BH_OPENCL_VOLATILE bohrium_release
fi
