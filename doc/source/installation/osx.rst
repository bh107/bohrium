Mac OSX
-------

The following explains how to get going on Mac OSX. Bohrium and NumPy is not compatible with the Python interpreter included with OSX. We recommend that you use Python version 2.7 from the `MacPorts <http://www.macports.org>`_ project. Furthermore, MacPorts have all packages that are needed to compile and install Bohrium.

You need to install the `Xcode Developer Tools <https://developer.apple.com/xcode/>`_, which is found in the App Store.

You also need the following packages from MacPorts::

  sudo port install python27 cmakei py27-cheetah

If you also want to build the Mono libraries (only required for the C# NumCIL package), you also need the Mono package::

  sudo port install mono

.. note:: The Mono version found on the `Mono homepage <http://www.mono-project.com/Main_Page>`_ does not support 64bit execution, and will not work with a normal build. You need to build a 32 bit version of Bohrium if you want to use the official Mono binaries.

Download and extract the current version (v0.2) (not recommended on OSX)::

  wget https://bitbucket.org/bohrium/bohrium/downloads/bohrium-v0.2.tgz
  tar -xzf bohrium-v0.2.tgz

As Bohrium is still under active development you want to build the current development copy, instead of using the tar-ball::

  git clone https://bitbucket.org/bohrium/bohrium.git
  cd bohrium

Make sure your system compiler is the one provided by Xcode, you can run the following command to verify that your compiler is the Apple version of clang::

  > gcc --version
  Configured with: --prefix=/Applications/Xcode.app/Contents/Developer/usr --with-gxx-include-dir=/usr/include/c++/4.2.1
  Apple LLVM version 5.0 (clang-500.2.79) (based on LLVM 3.3svn)
  Target: x86_64-apple-darwin13.0.0
  Thread model: posix

When building and installing Bohrium we need to specify the newly installed Python interpreter. In this case we use Python version 2.7::

  cmake .. -DPYTHON=python2.7

.. note:: If you want to make a system-wide installation, run the make install command with sudo.
          If you run the install command as a normal user, it will install all files to ``~/.local``.
          If you run the install command with sudo, it will install all files to ``/opt/bohrium``.

Since version 0.2 of Bohrium uses JIT compilation, you also need to edit your Bohrium configuration file, which is normally found in ``~/.bohrium/config.ini``. Find the section named ``cpu`` and edit the compiler command::

  [cpu]
  type = ve
  compiler_cmd="clang -arch x86_64 -I~/.local/include -lm -O3 -fPIC -std=c99 -x c -shared - -o "

This change is required to ensure that the kernels are compiled as 64-bit kernels. If you need 32-bit kernels, you can change  ``-arch x86_64`` to ``-arch i386``. If you have strange errors, it may be because you have invalid kernels in the cache directory. Issue the following command to clean it after changing the ``compiler_cmd``:

  rm -rf ~/.local/cpu/objects/*

.. note:: If you update your installation, you may want to delete your ``~/.bohrium/config.ini`` file and run the install command again. Remember to edit the compiler command again if you do this.

Python / NumPy
~~~~~~~~~~~~~~
You need to set the ``PYTHONPATH`` and the ``DYLD_LIBRARY_PATH`` environment variables.
The ``PYTHONPATH`` should include the path to the newly installed Bohrium Python module. This will also make sure that Python uses the NumPy module included in Bohrium::

  export PYTHONPATH=<install dir>/lib/python<python version>/site-packages:$PYTHONPATH
  #Example
  export PYTHONPATH=~/.local/lib/python2.7/site-packages:$PYTHONPATH

The ``DYLD_LIBRARY_PATH`` should include the path to the installation directory::

  export DYLD_LIBRARY_PATH=<install dir>:$DYLD_LIBRARY_PATH
  #Example
  export DYLD_LIBRARY_PATH=~/.local:$DYLD_LIBRARY_PATH

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
   gacutil -i ~/.local/NumCIL.dll
   gacutil -i ~/.local/NumCIL.Unsafe.dll
   gacutil -i ~/.local/NumCIL.Bohrium.dll

To use the Bohrium extensions, you need to make sure the DYLD_LIBRARY_PATH is also set::

  export DYLD_LIBRARY_PATH=<install dir>:$LD_LIBRARY_PATH
  #Example
  export DYLD_LIBRARY_PATH=~/.local:$LD_LIBRARY_PATH

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

