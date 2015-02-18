Linux
=====

The following instruct you on how to get going on the Ubuntu Linux distribution. There should however only be slight differences to other distributions such as which command to execute to install software packages.

Install From Debian Package
---------------------------

To install Bohrium on Ubuntu simply add the nightly build repository to your system::

  sudo add-apt-repository ppa:bohrium/nightly

And then install the package::

  sudo apt-get update
  sudo apt-get install bohrium

Now the basic installation should work. Try running the NumPy test suite::

  python /usr/share/bohrium/test/numpy/numpytest.py

And you should see a result similar to this::

    *** Testing the equivalency of Bohrium-NumPy and NumPy ***
    Testing test_primitives.py/bh_opcodes/ufunc
    Testing test_primitives.py/numpy_ufunc/ufunc
    Testing test_specials.py/doubletranspose/doubletranspose
    Testing test_specials.py/largedim/largedim
    Testing test_array_create.py/array_create/zeros
    Testing test_benchmarks.py/gameoflife/gameoflife
    Testing test_benchmarks.py/jacobi/jacobi
    Testing test_benchmarks.py/jacobi_stencil/jacobi_stencil
    Testing test_benchmarks.py/shallow_water/shallow_water
    Testing test_matmul.py/matmul/dot
    Testing test_matmul.py/matmul/matmul
    Testing test_types.py/different_inputs/typecast
    Testing test_reduce.py/reduce/reduce
    Testing test_reduce.py/reduce1D/reduce
    Testing test_views.py/diagonal/diagonal
    Testing test_views.py/flatten/flatten
    Testing test_sor.py/sor/sor
    ************************ Finish ************************

NumCIL (.NET) Support
~~~~~~~~~~~~~~~~~~~~~

In order to use NumCIL as the Bohrium frontend install the Bohrium NumCIL package::

    sudo apt-get install bohrium-numcil

GPU Support
~~~~~~~~~~~

In order to utilize GPUs you need an OpenCL 1.2 compatible graphics card and the Bohrium GPU package::

  sudo apt-get install bohrium-gpu

.. note:: On Nvidia Optimus architectures, remember to install and use bumblebee (``optirun``) when calling Bohrium.


Cluster Support
~~~~~~~~~~~~~~~

In order to utilize a Cluster of machines you must choose between the two supported MPI libraries::

  sudo apt-get install bohrium-openmpi
                or
  sudo apt-get install bohrium-mpich

Now execute using MPI::

  mpiexec -np 1 <user application> : -np 3 /usr/bin/bh_vem_cluster_slave

Where one process executes the user application and multiple processes executes the slave binary.

For example, the following utilize eight cluster nodes::

  mpiexec -np 1 python /usr/share/bohrium/test/numpy/numpytest.py : -np 7 /usr/bin/bh_vem_cluster_slave

When using OpenMPI you might have to set ``export LD_PRELOAD=/usr/lib/libmpi.so``.

.. warning:: The cluster engine is in a significantly less developed state than both the CPU and GPU engine.


Install From Source Package
---------------------------

Visit https://bitbucket.org/bohrium/bohrium/downloads and download a specific tarball release or the whole repository. Then build and install Bohrium as described in the following subsections.

.. note:: Currently, no stable version of Bohrium has been released thus only the whole repository is available for download: https://bitbucket.org/bohrium/bohrium/get/master.tgz.

Python / NumPy
~~~~~~~~~~~~~~

You need to install all packages required to build NumPy::

  sudo apt-get build-dep python-numpy

And some additional packages::

  sudo apt-get install python-numpy swig python-cheetah cmake libboost-serialization-dev cython libhwloc-dev libctemplate-dev

Build and install::

  wget https://bitbucket.org/bohrium/bohrium/get/master.tgz
  tar -xzf master.tgz
  cd bohrium-bohrium-<hash>
  mkdir build
  cd build
  cmake .. -DCMAKE_INSTALL_PREFIX=<path to install directory>
  make
  make install

.. note:: The default install directory is ~/.local

.. note:: To compile to a custom Python (with valgrind debug support for example), set ``-DPYTHON_EXECUTABLE=<custom python binary> -DPY_SCRIPT=python``.

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

  python test/python/numpytest.py

And you should see a result similar to this::

    *** Testing the equivalency of Bohrium-NumPy and NumPy ***
    Testing test_primitives.py/bh_opcodes/ufunc
    Testing test_primitives.py/numpy_ufunc/ufunc
    Testing test_specials.py/doubletranspose/doubletranspose
    Testing test_specials.py/largedim/largedim
    Testing test_array_create.py/array_create/zeros
    Testing test_benchmarks.py/gameoflife/gameoflife
    Testing test_benchmarks.py/jacobi/jacobi
    Testing test_benchmarks.py/jacobi_stencil/jacobi_stencil
    Testing test_benchmarks.py/shallow_water/shallow_water
    Testing test_matmul.py/matmul/dot
    Testing test_matmul.py/matmul/matmul
    Testing test_types.py/different_inputs/typecast
    Testing test_reduce.py/reduce/reduce
    Testing test_reduce.py/reduce1D/reduce
    Testing test_views.py/diagonal/diagonal
    Testing test_views.py/flatten/flatten
    Testing test_sor.py/sor/sor
    ************************ Finish ************************

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


MPI / Cluster Engine
~~~~~~~~~~~~~~~~~~~~

In order to utilize a computer clusters, you need to install mpich2 or OpenMPI before building Bohrium::

  sudo apt-get install mpich2 libmpich2-dev
                    or
  sudo apt-get install libopenmpi-dev openmpi-bin

And execute using mpi::

  mpiexec -np 1 <user application> : -np 3 <install dir>/bin/bh_vem_cluster_slave

Where one process executes the user application and multiple processes executes the slave binary from the installation directory.

For example, the following utilize eight cluster nodes::

  mpiexec -np 1 python numpytest.py : -np 7 ~/.local/bin/bh_vem_cluster_slave

When using OpenMPI you might have to set ``export LD_PRELOAD=/usr/lib/libmpi.so``.


.. warning:: The cluster engine is in a significantly less developed state than both the CPU and GPU engine.

