Mac OSX
-------

The following explains how to get going on Mac OSX. Bohrium and NumPy is not compatible with the Python interpreter included with OSX. We recommend that you use Python version 2.7 from the `MacPorts <http://www.macports.org>`_ project. Furthermore, MacPorts have all packages that are needed to compile and install Bohrium.

You need to install the `Xcode Developer Tools <https://developer.apple.com/technologies/tools/>`_ from Apple and the following packages from MacPorts::

 sudo port install python27

If you also want to build the Mono libraries, you also need the Mono package::

   sudo port install mono

.. note:: The Mono version found on the `Mono homepage <http://www.mono-project.com/Main_Page>`_ does not support 64bit execution, and will not work with a normal build. You need to build a 32 bit version of Bohrium if you want to use the official Mono binaries.

Download and extract the source code::

  git clone https://bitbucket.org/bohrium/bohrium.git
  cd bohrium
  git submodule init
  git submodule update

When building and install Bohrium we need to specify the newly installed Python interpreter. In this case we use Python version 2.7::

  make BH_PYTHON=python2.7
  make install BH_PYTHON=python2.7

.. note:: The installation will prompt you for the installation path.
          The default path is ``/opt/bohrium`` which requires root permissions. Hence, if you do not have root access use a installation path to inside your home directory.

Python / NumPy
~~~~~~~~~~~~~~
You need to set the ``PYTHONPATH`` and the ``DYLD_LIBRARY_PATH`` environment variables.
The ``PYTHONPATH`` should include the path to the newly installed Bohrium Python module. This will also make sure that Python uses the NumPy module included in Bohrium::

  export PYTHONPATH=<install dir>/lib/python<python version>/site-packages:$PYTHONPATH
  #Example
  export PYTHONPATH=/opt/bohrium/lib/python2.7/site-packages:$PYTHONPATH

The ``DYLD_LIBRARY_PATH`` should include the path to the installation directory::

  export DYLD_LIBRARY_PATH=<install dir>:$DYLD_LIBRARY_PATH
  #Example
  export DYLD_LIBRARY_PATH=/opt/bohrium:$DYLD_LIBRARY_PATH

Now the basic installation should work. Try running the NumPy test suite::

  python2.7 test/numpy/numpytest.py

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
The NumCIL libraries are installed in your install dir, together with the documentation. You can reference the libraries from here, or register them in the GAC::

   gacutil -i <install dir>/NumCIL.dll
   gacutil -i <install dir>/NumCIL.Unsafe.dll
   gacutil -i <install dir>/NumCIL.Bohrium.dll
   #Example
   gacutil -i /opt/bohrium/NumCIL.dll
   gacutil -i /opt/bohrium/NumCIL.Unsafe.dll
   gacutil -i /opt/bohrium/NumCIL.Bohrium.dll

To use the Bohrium extensions, you need to make sure the DYLD_LIBRARY_PATH is also set::

  export DYLD_LIBRARY_PATH=<install dir>:$LD_LIBRARY_PATH
  #Example
  export DYLD_LIBRARY_PATH=/opt/bohrium:$LD_LIBRARY_PATH

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

