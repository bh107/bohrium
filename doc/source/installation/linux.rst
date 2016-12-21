Linux
=====

The following instruct you on how to get going on the Ubuntu Linux distribution. There should however only be slight differences to other distributions such as which command to execute to install software packages.

Install From Ubuntu Package
---------------------------

To install Bohrium on Ubuntu simply add the nightly build repository to your system::

  sudo add-apt-repository ppa:bohrium/nightly

And then install the package::

  sudo apt-get update
  sudo apt-get install bohrium
  
And if you want Python v3 support::

  sudo apt-get install bohrium3

Now the basic installation should work. Try running the NumPy test suite::

  python /usr/share/bohrium/test/python/run.py /usr/share/bohrium/test/python/tests/test_*.py

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

Visualizer (matplotlib alternative)
~~~~~~~~~~~~~~~~~~~~~

In order to use the Bohrium visualizer install the Bohrium Visualizer package::

    sudo apt-get install bohrium-visualizer

NumCIL (.NET) Support
~~~~~~~~~~~~~~~~~~~~~

In order to use NumCIL as the Bohrium frontend install the Bohrium NumCIL package::

    sudo apt-get install bohrium-numcil

GPU Support
~~~~~~~~~~~

In order to utilize GPUs you need an OpenCL 1.2 compatible graphics card and the Bohrium GPU package::

  sudo apt-get install bohrium-opencl

.. note:: On Nvidia Optimus architectures, remember to install and use bumblebee (``optirun``) when calling Bohrium.


.. Cluster Support
.. ~~~~~~~~~~~~~~~
..
.. In order to utilize a Cluster of machines you must choose between the two supported MPI libraries::
..
..   sudo apt-get install bohrium-openmpi
..                 or
..   sudo apt-get install bohrium-mpich
..
.. Now execute using MPI::
..
..   mpiexec -np 1 <user application> : -np 3 /usr/bin/bh_vem_cluster_slave
..
.. Where one process executes the user application and multiple processes executes the slave binary.
..
.. For example, the following utilize eight cluster nodes::
..
..   mpiexec -np 1 python /usr/share/bohrium/test/numpy/numpytest.py : -np 7 /usr/bin/bh_vem_cluster_slave
..
.. When using OpenMPI you might have to set ``export LD_PRELOAD=/usr/lib/libmpi.so``.
..
.. .. warning:: The cluster engine is in a significantly less developed state than both the CPU and GPU engine.


Install From Source Package
---------------------------

Visit https://github.com/bh107/bohrium/downloads and download a specific tarball release or the whole repository. Then build and install Bohrium as described in the following subsections.

.. note:: Currently, no stable version of Bohrium has been released thus only the whole repository is available for download: https://github.com/bh107/bohrium/archive/master.zip

Python / NumPy
~~~~~~~~~~~~~~

You need to install all packages required to build NumPy::

  sudo apt-get build-dep python-numpy

And some additional packages::

  sudo apt-get install python-numpy python-dev swig cmake unzip cython libhwloc-dev libboost-filesystem-dev libboost-serialization-dev libboost-regex-dev  zlib1g-dev

And for python v3 support::
  
  sudo apt-get python3-dev python3-numpy python3-dev cython3

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

.. note:: The default install directory is ~/.local

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

  python test/python/run.py  test/python/tests/test_*.py

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


C / C++
~~~~~~~

See the installation process for :ref:`Python / NumPy <numpy_installation>`, the C and C++ bridge requires no additional tasks.


Mono / .NET
~~~~~~~~~~~

In addition to the installation process for :ref:`Python / NumPy <numpy_installation>`, the .NET bridge requires Mono::

  sudo apt-get install mono-devel
  #This minimal version should work too:
  #sudo apt-get install mono-xbuild mono-dmcs libmono2.0-cil

Build and install::

  cd <path to unpacked source directory>
  mkdir build
  cd build
  cmake .. -DCMAKE_INSTALL_PREFIX=<path to install directory>
  make
  make install

.. note:: The default install directory is ~/.local

The NumCIL libraries are installed in your install dir, together with the documentation. You can reference the libraries from here, or register them in the GAC::

   gacutil -i <install dir>/NumCIL.dll
   gacutil -i <install dir>/NumCIL.Unsafe.dll
   gacutil -i <install dir>/NumCIL.Bohrium.dll
   #Example
   gacutil -i /opt/bohrium/NumCIL.dll
   gacutil -i /opt/bohrium/NumCIL.Unsafe.dll
   gacutil -i /opt/bohrium/NumCIL.Bohrium.dll

You can now try an example and test the installation::

  xbuild /property:Configuration=Release test/CIL/Unittest.sln
  mono test/CIL/UnitTest/bin/Release/UnitTest.exe

And you should see a result similar to this::

   Running basic tests
   Basic tests: 0,098881
   Running Lookup tests
   Lookup tests: 0,00813
   ...
   Running benchmark tests - Bohrium
   benchmark tests: 0,44233


OpenCL / GPU Engine
~~~~~~~~~~~~~~~~~~~

The GPU vector engine requires OpenCL compatible hardware as well as functioning drivers.
Configuring your GPU with you operating system is out of scope of this documentation.

Assuming that your GPU-hardware is functioning correctly you need to install an OpenCL SDK and some additional packages before building Bohrium::

  sudo apt-get install opencl-dev libopencl1 libgl-dev

You should now have everything you need to utilize the GPU engine.


.. MPI / Cluster Engine
.. ~~~~~~~~~~~~~~~~~~~~
..
.. In order to utilize a computer clusters, you need to install mpich2 or OpenMPI before building Bohrium::
..
..   sudo apt-get install mpich2 libmpich2-dev
..                     or
..   sudo apt-get install libopenmpi-dev openmpi-bin
..
.. And execute using mpi::
..
..   mpiexec -np 1 <user application> : -np 3 <install dir>/bh_vem_cluster_slave
..
.. Where one process executes the user application and multiple processes executes the slave binary from the installation directory.
..
.. For example, the following utilize eight cluster nodes::
..
..   mpiexec -np 1 python numpytest.py : -np 7 .local/bh_vem_cluster_slave
..
.. When using OpenMPI you might have to set ``export LD_PRELOAD=/usr/lib/libmpi.so``.
..
..
.. .. warning:: The cluster engine is in a significantly less developed state than both the CPU and GPU engine.
