Linux
=====

.. highlight:: ruby

PyPI Package
------------

If you use Bohrium through Python, we strongly recommend to install Bohrium through `pypi <https://pypi.python.org/pypi>`_, which will include BLAS, LAPACK, OpenCV, and OpenCL support::

    pip install --user bohrium

.. note:: On linux, Bohrium requires ``gcc`` in ``$PATH``. E.g. on Ubuntu install the build-essential package: ``sudo apt install build-essential``.

Ubuntu
------

On Ubuntu you can use apt-get::

    sudo add-apt-repository ppa:bohrium/nightly
    sudo apt-get update
    sudo apt-get install bohrium
    # Optionals
    sudo apt-get install bohrium-opencl # GPU support
    sudo apt-get install bohrium-visualizer # data visualizing
    sudo apt-get install bohrium3 # Python3 support

Anaconda
--------

To use Anaconda, simply install the Bohrium PyPI package in an environment::

    # Activate the environment where you want to install Bohrium:
    source activate my_env
    # Install Bohrium using pip
    pip install bohrium

.. note:: Bohrium requires ``gcc`` in ``$PATH``. E.g. on Ubuntu install the build-essential package: ``sudo apt install build-essential``.

Install From Source Package
---------------------------

Visit Bohrium on github.com and download the latest release: https://github.com/bh107/bohrium/releases/latest. Then build and install Bohrium as described in the following subsections.

You need to install all packages required to build NumPy::

  sudo apt-get build-dep python-numpy

And some additional packages::

  sudo apt-get install python-numpy python-dev swig cmake unzip cython libhwloc-dev libboost-filesystem-dev libboost-serialization-dev libboost-regex-dev zlib1g-dev libsigsegv-dev

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

The ``PYTHONPATH`` should include the path to the newly installed Bohrium Python module::

    export PYTHONPATH="<install dir>/lib/python<python version>/site-packages:$PYTHONPATH"

Check Your Installation
-----------------------

Check installation by printing the current runtime stack::

    python -m bohrium --info
