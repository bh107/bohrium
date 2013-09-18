Linux
-----

The following instruct you on how to get going on the Ubuntu Linux distribution. There should however only be slight differences to other distributions such as which command to execute to install software packages.


Python / NumPy
~~~~~~~~~~~~~~

You need to install all packages required to build NumPy::

  sudo apt-get build-dep python-numpy

And some additional packages::

  sudo apt-get install g++ python-dev python-pip python-cheetah python-sphinx doxygen libmpich2-dev git
  sudo pip install breathe numpydoc

Download and extract the source code::

  git clone https://bitbucket.org/bohrium/bohrium.git
  cd bohrium
  git submodule init
  git submodule update

Build and install::

  make
  make install

.. note:: The installation will prompt you for the installation path.

.. note:: To compile to a custom Python (with valgrind debug support for example), set the make variable, BH_PYTHON, naming the binary of your custom compiled Python.

Finally, you need to set the ``PYTHONPATH`` and the ``LD_LIBRARY_PATH`` environment variables.
The ``PYTHONPATH`` should include the path to the newly installed Bohrium Python module. This will also make sure that Python uses the NumPy module included in Bohrium::

  export PYTHONPATH=<install dir>/lib/python<python version>/site-packages:$PYTHONPATH
  #Example
  export PYTHONPATH=/opt/bohrium/lib/python2.7/site-packages:$PYTHONPATH

The ``LD_LIBRARY_PATH`` should include the path to the installation directory::

  export LD_LIBRARY_PATH="<install dir>:$LD_LIBRARY_PATH"
  #Example
  export LD_LIBRARY_PATH="$HOME/.local:$LD_LIBRARY_PATH"

Now the basic installation should work. Try running the NumPy test suite::

  python test/numpy/numpytest.py

And you should see a result similar to this::

  *** Testing the equivalency of Bohrium-NumPy and NumPy ***
  Testing test_array_create.py/array_create/zeros
  Testing test_sor.py/sor/sor
  Testing test_primitives.py/bh_opcodes/ufunc
  Testing test_primitives.py/numpy_ufunc/ufunc
  Testing test_reduce.py/reduce/reduce
  Testing test_benchmarks.py/gameoflife/gameoflife
  Testing test_benchmarks.py/jacobi/jacobi
  Testing test_benchmarks.py/jacobi_stencil/jacobi_stencil
  Testing test_benchmarks.py/shallow_water/shallow_water
  Testing test_matmul.py/matmul/dot
  Testing test_matmul.py/matmul/matmul
  Testing test_views.py/diagonal/diagonal
  Testing test_views.py/flatten/flatten
  ************************ Finish ************************

Mono / .NET
~~~~~~~~~~~

You need to install some packages used by the build process::

  sudo apt-get install g++ python-dev python-pip python-cheetah python-sphinx doxygen libmpich2-dev git

The Mono libraries require some additional packages::

  sudo apt-get install mono-devel
  #This minimal version should work too:
  #sudo apt-get install mono-xbuild mono-dmcs libmono2.0-cil

Download and extract the source code::

  git clone https://bitbucket.org/bohrium/bohrium.git
  cd bohrium
  git submodule init
  git submodule update

Build and install::

  make
  make install

.. note:: The installation will prompt you for the installation path.

The NumCIL libraries are installed in your install dir, together with the documentation. You can reference the libraries from here, or register them in the GAC::

   gacutil -i <install dir>/NumCIL.dll
   gacutil -i <install dir>/NumCIL.Unsafe.dll
   gacutil -i <install dir>/NumCIL.Bohrium.dll
   #Example
   gacutil -i /opt/bohrium/NumCIL.dll
   gacutil -i /opt/bohrium/NumCIL.Unsafe.dll
   gacutil -i /opt/bohrium/NumCIL.Bohrium.dll

To use the Bohrium extensions, you need to make sure the LD_LIBRARY_PATH is also set::

  export LD_LIBRARY_PATH=<install dir>:$LD_LIBRARY_PATH
  #Example
  export LD_LIBRARY_PATH=/opt/bohrium:$LD_LIBRARY_PATH

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

C++
~~~

...

OpenCL / GPU Engine
~~~~~~~~~~~~~~~~~~~

The GPU vector engine requires OpenCL compatible hardware as well as functioning drivers.
Configuring your GPU with you operating system is out of scope of this documentation.

Assuming that your GPU-hardware is functioning correctly you need to install an OpenCL SDK and some additional packages.

**Packages**::

  sudo apt-get install -y rpm alien libnuma1

**SDK for OpenCL**

Go to http://software.intel.com/en-us/articles/vcsource-tools-opencl-sdk/ and download *Intel SDK for OpenCL 2012 -- Linux*.

The download-button is in the upper right corner next to select-box with the text *Select version...*.

The download area is hard to spot, so take a look at the red arrow on the picture below:

.. image:: opencl_download.png
   :scale: 50 %
   :alt: Download location.

Once downloaded, install the SDK with the following commands::

  tar zxf intel_sdk_for_ocl_applications_2012_x64.tgz
  fakeroot alien --to-deb intel_ocl_sdk_2012_x64.rpm
  sudo dpkg -i intel-ocl-sdk_2.0-31361_amd64.deb
  sudo ln -s /usr/lib64/libOpenCL.so /usr/lib/libOpenCL.so
  sudo ldconfig

You should now have everything you need to utilize the GPU engine.


MPI / Cluster Engine
~~~~~~~~~~~~~~~~~~~~

In order to utilize a computer clusters, you need to install mpich2::

  sudo apt-get install mpich2

And execute using mpi::

  mpiexec -np 1 <user application> : -np 3 <install dir>/bh_vem_cluster_slave

Where one process executes the user application and multiple processes executes the slave binary from the installation directory.

For example, the following utilize eight cluster nodes::

  mpiexec -np 1 python numpytest.py : -np 7 .local/bh_vem_cluster_slave

