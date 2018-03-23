Mac OS
======

.. highlight:: ruby

The following explains how to get going on Mac OS.

You need to install the `Xcode Developer Tools <https://developer.apple.com/xcode/>`_ package, which is found in the App Store.

PyPI Package
------------

If you use Bohrium through Python, we strongly recommend to install Bohrium through `pypi <https://pypi.python.org/pypi>`_, which will include BLAS, LAPACK, OpenCV, and OpenCL support::

    python -m pip install --user bohrium

Homebrew
~~~~~~~~

Start by `installing Homebrew as explained on their website <http://brew.sh/>`_ ::

  /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Then install Bohrium::

  pip install cython # This dependency cannot be installed via brew.
  brew tap bh107/bohrium
  brew tap homebrew/science # for clblas and the likes
  brew install bohrium # you can add additional options, see `brew info bohrium`

Install From Source Package
---------------------------

Start by `installing Homebrew as explained on their website <http://brew.sh/>`_ ::

  /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Install dependencies::

  brew install python
  brew install cmake
  brew install boost --with-icu4c
  brew install libsigsegv
  python3 -m pip install --user numpy cython twine

Visit Bohrium on github.com, download the latest release: https://github.com/bh107/bohrium/releases/latest or download `master`, and then build it::

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

The ``PYTHONPATH`` should include the path to the newly installed Bohrium Python module::

    export PYTHONPATH="<install dir>/lib/python<python version>/site-packages:$PYTHONPATH"

Check Your Installation
-----------------------

Check installation by printing the current runtime stack::

    python -m bohrium --info

