Linux
=====

On Ubuntu use apt-get::

    sudo add-apt-repository ppa:bohrium/nightly
    sudo apt-get update
    sudo apt-get install bohrium
    # Optionals
    sudo apt-get install bohrium-opencl # GPU support
    sudo apt-get install bohrium-visualizer # data visualizing
    sudo apt-get install bohrium3 # Python3 support

On other 64-bit Linux systems use `Anaconda <https://www.continuum.io/downloads>`_ (currently, no GPU support)::

    # Create a new environment 'bh' with the 'bohrium' package from the 'bohrium' channel:
    conda create -n bh -c bohrium bohrium
    # And source the new environment:
    source activate bh

Check installation by printing the current runtime stack::

    python -c "import bohrium as bh; print(bh.bh_info.runtime_info())"

And try running the test suite::

      # Installed through apt-get:
      BH_OPENMP_VOLATILE=true python /usr/share/bohrium/test/python/run.py /usr/share/bohrium/test/python/tests/test_*.py

      # Install through Anaconda:
      BH_OPENMP_VOLATILE=true python $CONDA_PREFIX/share/bohrium/test/python/run.py  $CONDA_PREFIX/share/bohrium/test/python/tests/test_*.py

And you should see a result similar to this::

      Testing test/python/tests/test_array_create.py/array_create/ones (0.84s) ✓
      Testing test/python/tests/test_array_create.py/array_create/random (3.11s) ✓
      Testing test/python/tests/test_array_create.py/array_create/zeros (0.72s) ✓
      Testing test/python/tests/test_array_manipulation.py/diagonal/diagonal (4.65s) ✓
      Testing test/python/tests/test_array_manipulation.py/diagonal_axis/diagonal (27.84s) ✓
      Testing test/python/tests/test_array_manipulation.py/flatten/flatten (6.15s) ✓
      Testing test/python/tests/test_array_manipulation.py/flatten/flatten_self (6.29s) ✓
      Testing test/python/tests/test_array_manipulation.py/flatten/ravel (5.94s) ✓
      Testing test/python/tests/test_array_manipulation.py/overlapping/add (0.35s) ✓
      Testing test/python/tests/test_array_manipulation.py/overlapping/identity (0.08s) ✓
      Testing test/python/tests/test_array_manipulation.py/transpose/doubletranspose (4.94s) ✓
      Testing test/python/tests/test_array_manipulation.py/transpose/transpose (1.97s) ✓
      Testing test/python/tests/test_emptiness.py/empty/add (0.00s) ✓
      <string>:1: RuntimeWarning: invalid value encountered in arccosh
      <string>:1: RuntimeWarning: invalid value encountered in arccosh
      Testing test/python/tests/test_primitives.py/bh_opcodes/ufunc (67.92s) ✓
      Testing test/python/tests/test_reduce.py/reduce_primitives/vector (3.21s) ✓
      Testing test/python/tests/test_reduce.py/reduce_sum/func (13.38s) ✓
      Testing test/python/tests/test_reduce.py/reduce_sum/method (9.81s) ✓
      Testing test/python/tests/test_reduce.py/reduce_views/reduce (20.55s) ✓

.. note:: We set ``BH_OPENMP_VOLATILE=true`` in order to avoid precision differences because of Intel's use of 80-bit floats internally.


Install From Source Package
---------------------------

Visit Bohrium on github.com and download the latest release: https://github.com/bh107/bohrium/releases/latest. Then build and install Bohrium as described in the following subsections.

You need to install all packages required to build NumPy::

  sudo apt-get build-dep python-numpy

And some additional packages::

  sudo apt-get install python-numpy python-dev swig cmake unzip cython libhwloc-dev libboost-filesystem-dev libboost-serialization-dev libboost-regex-dev zlib1g-dev

And for python v3 support::

  sudo apt-get install python3-dev python3-numpy python3-dev cython3

Packages for visualization::

  sudo apt-get install freeglut3 freeglut3-dev libxmu-dev libxi-dev

Build and install::

  wget https://github.com/bh107/bohrium/archive/master.zip
  unzip master.zip
  cd bohrium-master
  mkdir build
  cd build
  cmake .. -DCMAKE_INSTALL_PREFIX=<path to install directory>
  make
  make install

.. note:: The default install directory is ``~/.local``

.. note:: To compile to a custom Python (with valgrind debug support for example), set ``-DPYTHON_EXECUTABLE=<custom python binary>``.

Finally, you need to set the ``LD_LIBRARY_PATH`` environment variables and if you didn't install Bohrium in ``$HOME/.local/lib`` your need to set ``PYTHONPATH`` as well.

The ``LD_LIBRARY_PATH`` should include the path to the installation directory::

  export LD_LIBRARY_PATH="<install dir>:$LD_LIBRARY_PATH"
  #Example
  export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"


The ``PYTHONPATH`` should include the path to the newly installed Bohrium Python module.::

  export PYTHONPATH=<install dir>/lib/python<python version>/site-packages:$PYTHONPATH
  #Example
  export PYTHONPATH=/opt/bohrium/lib/python2.7/site-packages:$PYTHONPATH

Now the basic installation should work. Try running the NumPy test suite::

  BH_OPENMP_VOLATILE=true python test/python/run.py  test/python/tests/test_*.py

And you should see a result similar to this::

  Testing test/python/tests/test_array_create.py/array_create/ones (0.84s) ✓
  Testing test/python/tests/test_array_create.py/array_create/random (3.11s) ✓
  Testing test/python/tests/test_array_create.py/array_create/zeros (0.72s) ✓
  Testing test/python/tests/test_array_manipulation.py/diagonal/diagonal (4.65s) ✓
  Testing test/python/tests/test_array_manipulation.py/diagonal_axis/diagonal (27.84s) ✓
  Testing test/python/tests/test_array_manipulation.py/flatten/flatten (6.15s) ✓
  Testing test/python/tests/test_array_manipulation.py/flatten/flatten_self (6.29s) ✓
  Testing test/python/tests/test_array_manipulation.py/flatten/ravel (5.94s) ✓
  Testing test/python/tests/test_array_manipulation.py/overlapping/add (0.35s) ✓
  Testing test/python/tests/test_array_manipulation.py/overlapping/identity (0.08s) ✓
  Testing test/python/tests/test_array_manipulation.py/transpose/doubletranspose (4.94s) ✓
  Testing test/python/tests/test_array_manipulation.py/transpose/transpose (1.97s) ✓
  Testing test/python/tests/test_emptiness.py/empty/add (0.00s) ✓
  <string>:1: RuntimeWarning: invalid value encountered in arccosh
  <string>:1: RuntimeWarning: invalid value encountered in arccosh
  Testing test/python/tests/test_primitives.py/bh_opcodes/ufunc (67.92s) ✓
  Testing test/python/tests/test_reduce.py/reduce_primitives/vector (3.21s) ✓
  Testing test/python/tests/test_reduce.py/reduce_sum/func (13.38s) ✓
  Testing test/python/tests/test_reduce.py/reduce_sum/method (9.81s) ✓
  Testing test/python/tests/test_reduce.py/reduce_views/reduce (20.55s) ✓


.. note:: We set ``BH_OPENMP_VOLATILE=true`` in order to avoid precision differences because of Intel's use of 80-bit floats internally.
