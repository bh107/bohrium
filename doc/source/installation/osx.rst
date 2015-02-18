Mac OSX
-------

The following explains how to get going on Mac OSX. You need some extra packages to build Bohrium from source. All packages are available from Homebrew, MacPorts and Fink, so you can use your prefered package manager.

You need to install the `Xcode Developer Tools <https://developer.apple.com/xcode/>`_ package, which is found in the App Store.

Start with installing pip and then the required Python packages::

  sudo easy_install pip
  sudo pip install cheetah cython numpy

Then install the required pacakages using your favorite package manager.

If you are using Mac MacPorts::

  # System Packages
  sudo port install cmake swig boost
  # Mono package, not required
  sudo port install mono

If you are using Homebrew::

  # System Packages
  brew install cmake swig
  brew install boost --universal
  # Mono package, not required
  brew install mono

If you are using finkproject::

  # System Packages
  fink install cmake swig boost1.53.nopython
  # Mono package, not required
  fink install mono


.. note:: The Mono version found on the `Mono homepage <http://www.mono-project.com/Main_Page>`_ is 32bit and thus only supports up to 2GB memory.


As Bohrium is still under active development you want to build the current development copy, instead of using the tar-ball::

  git clone https://bitbucket.org/bohrium/bohrium.git
  cd bohrium

Make sure your system compiler is the one provided by Xcode, you can run the following command to verify that your compiler is the Apple version of clang::

  > gcc --version
  Configured with: --prefix=/Applications/Xcode.app/Contents/Developer/usr --with-gxx-include-dir=/usr/include/c++/4.2.1
  Apple LLVM version 5.0 (clang-500.2.79) (based on LLVM 3.3svn)
  Target: x86_64-apple-darwin13.0.0
  Thread model: posix

Bohrium uses CMake so everything is configured automatically::

  mkdir build
  cd build
  cmake ..
  make install
  cd ..

.. note:: If you want to make a system-wide installation, run the make install command with sudo.
          If you run the install command as a normal user, it will install all files to ``~/.local``.
          If you run the install command with sudo, it will install all files to ``/opt/bohrium``.

If you are upgrading from a previous version you should delete the config file and have a fresh one installed::

  rm ~/.bohrium/config.ini
  make install

If you have previously used Bohrium, issue the following command to clean the old install::

  rm -rf ~/.local/var/bohrium ~/.local/share/bohrium/

Python / NumPy
~~~~~~~~~~~~~~
You need to set the ``PYTHONPATH`` and the ``DYLD_LIBRARY_PATH`` environment variables.
The ``PYTHONPATH`` should include the path to the newly installed Bohrium Python module. This will also make sure that Python uses the NumPy module included in Bohrium::

  export PYTHONPATH=<install dir>/lib/python<python version>/site-packages:$PYTHONPATH
  #Example
  export PYTHONPATH=~/.local/lib/python2.7/site-packages:$PYTHONPATH

The ``DYLD_LIBRARY_PATH`` should include the path to the installation directory::

  export DYLD_LIBRARY_PATH=<install dir>/lib:$DYLD_LIBRARY_PATH
  #Example
  export DYLD_LIBRARY_PATH=~/.local/lib:$DYLD_LIBRARY_PATH

Now the basic installation should work. Try running the NumPy test suite::

  python test/python/numpytest.py

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

If you prefer to not mess with your system and instead use the commandline, you can execute the unittest like this::

  MONO_PATH=~/.local/lib/mono DYLD_LIBRARY_PATH=~/.local/lib mono ~/.local/share/bohrium/test/cil/bin/UnitTest.exe


