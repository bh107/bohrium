Developer Tools
===============

Tools of the trade::

  sudo apt-get install git valgrind g++

Valgrind and Python
-------------------

Valgrind is a great tool for memory debugging, memory leak detection, and profiling.
However, both Python and NumPy floods the valgrind output with memory errors - it is therefore necessary to use a debug and valgrind friendly version of Python::

  sudo apt-get build-dep python
  PV=2.7.3
  sudo mkdir /opt/python
  cd /tmp
  wget http://www.python.org/ftp/python/$PV/Python-$PV.tgz
  tar xf Python-$PV.tgz
  cd Python-$PV
  ./configure --with-pydebug --without-pymalloc --with-valgrind --prefix /opt/python
  sudo make install
  sudo ln -s /opt/python/bin/python /usr/bin/dython

Valgrind can be used to detect memory errors by invoking it with::

  valgrind --vex-iropt-precise-memory-exns=yes dython <SCRIPT_NAME>

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

Building and Installing
-----------------------

In addition to the tools described above, the following must be present::

  # Essential dependencies
  sudo apt-get install python-dev mpi-default-dev

  # Code and documentation generator-tools
  sudo apt-get install python-pip python-cheetah python-sphinx doxygen
  sudo pip install breathe

Get the source-code::

  git clone git@bitbucket.org:cphvb/cphvb-priv.git
  cd cphvb-priv
  git submodule init
  git submodule update

Build and install it::

  ./build.py install

.. note:: To compile to a custom Python (with valgrind debug support for example),
   set the $PYTHON variable naming the binary of your custom compiled Python::

     PYTHON=dython ./build.py install

Automated Build / Jenkins
-------------------------

https://wiki.jenkins-ci.org/display/JENKINS/Installing+Jenkins+on+Ubuntu
Setup jenkins::

  wget -q -O - http://pkg.jenkins-ci.org/debian/jenkins-ci.org.key | sudo apt-key add -
  sudo sh -c 'echo deb http://pkg.jenkins-ci.org/debian binary/ > /etc/apt/sources.list.d/jenkins.list'
  sudo apt-get update
  sudo apt-get install jenkins

Then configure it via web-interface.


Writing Documentation
---------------------

The documentation is written in Sphinx.

You will need the following to write/build the documentation::

  sudo apt-get install doxygen python-sphinx python-docutils python-setuptools

As well as a python-package "breathe" for integrating doxygen-docs with Sphinx::

  sudo easy_install breathe

Overview of the documentatation files::

  cphvb/doc                 # Root folder of the documentation.
  cphvb/doc/source          # Write / Edit the documentation here.
  cphvb/doc/build           # Documentation is "rendered" and stored here.
  cphvb/doc/Makefile        # This file instructs Sphinx on how to "render" the documentation.
  cphvb/doc/make.bat        # ---- || ----, on Windows
  cphvb/doc/deploy_doc.sh   # This script pushes the rendered docs to http://cphvb.bitbucket.org

Most used commands
~~~~~~~~~~~~~~~~~~

These commands assume that your current working dir is cphvb/doc.

Initiate doxygen::
 
  make doxy

Render a html version of the docs::

  make html

Push the html to http://cphvb.bitbucket.org, this command assumes that you have write-access to the doc-repos on bitbucket::

  ./deploy_doc.sh

Create doxygen docs from source-code::

  doxygen Doxyfile

The docs still needs a neat way to integrate a full API-documentation of the cphVB core, managers and engines.
