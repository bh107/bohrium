.. _developer_tools:

Tools
=====

Valgrind, GDB, and Python
-------------------------

Valgrind is a great tool for memory debugging, memory leak detection, and profiling.
However, both Python and NumPy floods the valgrind output with memory errors - it is therefore necessary to use a debug and valgrind friendly version of Python and NumPy::

  sudo apt-get build-dep python
  sudo apt-get install zlib1g-dev valgrind

  mkdir python_debug_env
  cd python_debug_env
  export INSTALL_DIR=$PWD

  # Build and install Python:
  export VERSION=2.7.11
  wget http://www.python.org/ftp/python/$VERSION/Python-$VERSION.tgz
  tar -xzf Python-$VERSION.tgz
  cd Python-$VERSION
  ./configure --with-pydebug --without-pymalloc --with-valgrind --prefix=$INSTALL_DIR
  make install
  sudo ln -s $PWD/python-gdb.py /usr/bin/python-gdb.py
  sudo ln -s $INSTALL_DIR/bin/python /usr/bin/dython
  cd ..
  rm Python-$VERSION.tgz

  # Build and install Cython
  export VERSION=0.24
  wget http://cython.org/release/Cython-$VERSION.tar.gz
  tar -xzf Cython-$VERSION.tar.gz
  cd Cython-$VERSION
  dython setup.py install
  cd ..
  rm Cython-$VERSION.tar.gz

  export VERSION=21.1.0
  wget https://pypi.python.org/packages/f0/32/99ead2d74cba43bd59aa213e9c6e8212a9d3ed07805bb66b8bf9affbb541/setuptools-$VERSION.tar.gz#md5=8fd8bdbf05c286063e1052be20a5bd98
  tar -xzf setuptools-$VERSION.tar.gz
  cd setuptools-$VERSION
  dython setup.py install
  cd ..
  rm setuptools-$VERSION.tar.gz

  # Build and install NumPy
  export VERSION=1.11.0
  wget  https://github.com/numpy/numpy/archive/v$VERSION.tar.gz
  tar -xzf v$VERSION.tar.gz
  cd numpy-$VERSION
  dython setup.py install
  cd ..
  rm v$VERSION.tar.gz

Build Bohrium with custom Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build and install Bohrium (with some components deactivated)::

  unzip master.zip
  cd bohrium-master
  mkdir build
  cd build
  cmake .. -DPYTHON_EXECUTABLE=/usr/bin/dython -DEXT_FFTW=OFF -DEXT_VISUALIZER=OFF -DVEM_PROXY=OFF -DVE_GPU=OFF  -DBRIDGE_NUMCIL=OFF -DTEST_CIL=OFF
  make
  make install
  cd ..
  rm master.zip

Most Used Commands
``````````````````

**GDB**

GDB supports some helpful Python commands (https://docs.python.org/devguide/gdb.html). To activate, ``source`` the ``python-gdb.py`` file within GDB::

  source /usr/bin/python-gdb.py

Then you can use Python specific GDB commands such as ``py-list`` or ``py-bt``.


**Valgrind**

Valgrind can be used to detect memory errors by invoking it with::

  valgrind --suppressions=<path to bohrium>/misc/valgrind.supp dython <SCRIPT_NAME>

Narrowing the valgrind analysis, add the following to your source code::

  #include <valgrind/callgrind.h>
  ... your code ...
  CALLGRIND_START_INSTRUMENTATION;
  ... your code ...
  CALLGRIND_STOP_INSTRUMENTATION;
  CALLGRIND_DUMP_STATS;

Then run valgrind with the flag::

  --instr-atstart=no

Invoking valgrind to determine cache-utilization::

  --tool=callgrind --simulate-cache=yes <PROG> <PROG_PARAM>

Cluster VEM (MPI)
~~~~~~~~~~~~~~~~~

In order to use MPI with valgrind, the MPI implementation needs to be compiled with PIC and no-dlopen flag. E.g, `OpenMPI <http://www.open-mpi.org/>`_ could be installed as follows::

  wget http://www.open-mpi.org/software/ompi/v1.6/downloads/openmpi-1.6.5.tar.gz
  cd tar -xzf openmpi-1.6.5.tar.gz
  cd openmpi-1.6.5
  ./configure --with-pic --disable-dlopen --prefix=/opt/openmpi
  make
  sudo make install

And then executed using valgrind::

  export LD_LIBRARY_PATH=/opt/openmpi/lib/:$LD_LIBRARY_PATH
  export PATH=/opt/openmpi/bin:$PATH
  mpiexec -np 1 valgrind dython test/numpy/numpytest.py : -np 1 valgrind ~/.local/bh_vem_cluster_slave




Writing Documentation
---------------------

The documentation is written in `Sphinx <http://sphinx.pocoo.org/>`_.

You will need the following to write/build the documentation::

  sudo apt-get install doxygen python-sphinx python-docutils python-setuptools

As well as a python-packages **breathe** and **numpydoc** for integrating doxygen-docs with Sphinx::

  sudo easy_install breathe numpydoc

Overview of the documentation files::

  bohrium/doc                 # Root folder of the documentation.
  bohrium/doc/source          # Write / Edit the documentation here.
  bohrium/doc/build           # Documentation is "rendered" and stored here.
  bohrium/doc/Makefile        # This file instructs Sphinx on how to "render" the documentation.
  bohrium/doc/make.bat        # ---- || ----, on Windows
  bohrium/doc/deploy_doc.sh   # This script pushes the rendered docs to http://bohrium.bitbucket.org.

Most used commands
~~~~~~~~~~~~~~~~~~

These commands assume that your current working dir is **bohrium/doc**.

Initiate doxygen::

  make doxy

Render a html version of the docs::

  make html

Push the html-rendered docs to http://bohrium.bitbucket.org, this command assumes that you have write-access to the doc-repos on bitbucket::

  make deploy

The docs still needs a neat way to integrate a full API-documentation of the Bohrium core, managers and engines.

Continuous Integration
----------------------

Currently we use both a privately hosted `Jenkins <https://bohrium.erda.dk/jenkins/>`_ server as well as `Travis <https://travis-ci.org/bh107/bohrium>`_ for our CI.

Setup jenkins::

  wget -q -O - http://pkg.jenkins-ci.org/debian/jenkins-ci.org.key | sudo apt-key add -
  sudo sh -c 'echo deb http://pkg.jenkins-ci.org/debian binary/ > /etc/apt/sources.list.d/jenkins.list'
  sudo apt-get update
  sudo apt-get install jenkins

Then configure it via the web interface.
