Mac OS
======

.. highlight:: ruby

The following explains how to get going on Mac OS.

You need to install the `Xcode Developer Tools <https://developer.apple.com/xcode/>`_ package, which is found in the App Store.

.. note:: You might have to manually install some extra header files by running ```sudo installer -pkg /Library/Developer/CommandLineTools/Package/macOS_SDK_headers_for_macOS_10.14.pkg -target /``` where ```10.14``` is your current version (`more info <https://apple.stackexchange.com/questions/337940/why-is-usr-include-missing-i-have-xcode-and-command-line-tools-installed-moja>`_).

PyPI Package
------------

If you use Bohrium through Python, we strongly recommend to install Bohrium through `pypi <https://pypi.python.org/pypi>`_, which will include BLAS, LAPACK, OpenCV, and OpenCL support::

    python -m pip install --user bohrium
   
.. note:: If you get an error message saying that no package match your criteria it is properly because you are using a Python version for which `no package exist https://pypi.org/project/bohrium-api/#files`_ .  Please contact us and we will build a package using your specific Python version.
    

Install From Source Package
---------------------------

Start by `installing Homebrew as explained on their website <http://brew.sh/>`_ ::

  /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Install dependencies::

  brew install python
  brew install cmake
  brew install boost --with-icu4c
  brew install libsigsegv
  python3 -m pip install --user numpy cython twine gcc7

Visit Bohrium on github.com, download the latest release: https://github.com/bh107/bohrium/releases/latest or download `master`, and then build it::

  wget https://github.com/bh107/bohrium/archive/master.zip
  unzip master.zip
  cd bohrium-master
  mkdir build
  cd build
  export PATH="$(brew --prefix)/bin:/usr/local/opt/llvm/bin:/usr/local/opt/opencv3/bin:$PATH"
  export CC="clang"
  export CXX="clang++"
  export C_INCLUDE_PATH=$(llvm-config --includedir)
  export CPLUS_INCLUDE_PATH=$(llvm-config --includedir)
  export LIBRARY_PATH=$(llvm-config --libdir):$LIBRARY_PATH
  cmake .. -DCMAKE_INSTALL_PREFIX=<path to install directory>
  make
  make install

.. note:: The default install directory is ``~/.local``

.. note:: To compile to a custom Python (with valgrind debug support for example), set ``-DPYTHON_EXECUTABLE=<custom python binary>``.

Finally, you need to set the ``DYLD_LIBRARY_PATH`` and ``LIBRARY_PATH`` environment variables and if you didn't install Bohrium in ``$HOME/.local/lib`` your need to set ``PYTHONPATH`` as well.

The ``DYLD_LIBRARY_PATH`` and ``LIBRARY_PATH`` should include the path to the installation directory::

    export DYLD_LIBRARY_PATH="<install dir>:$DYLD_LIBRARY_PATH"
    export LIBRARY_PATH="<install dir>:$LIBRARY_PATH"

The ``PYTHONPATH`` should include the path to the newly installed Bohrium Python module::

    export PYTHONPATH="<install dir>/lib/python<python version>/site-packages:$PYTHONPATH"

Check Your Installation
-----------------------

Check installation by printing the current runtime stack::

    python -m bohrium --info

