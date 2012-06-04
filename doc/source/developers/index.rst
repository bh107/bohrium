cphVB Developer Guide
=======================

...

Tools and Environment
---------------------

Tools of the trade::

  sudo apt-get install git valgrind g++

Valgrind and Python
~~~~~~~~~~~~~~~~~~~

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

cphVB in short
--------------

...

Conventions
~~~~~~~~~~~

...

Core
~~~~

Instructions
~~~~~~~~~~~~

Think of instructions as good old MIPS-assembly::

  ADD $d, $s, $t

But instead of the operands $d, $s, and $t being registers they are either arrays, scalars or constants. Instructions are encapsulations of the most basic executable operation. For correct execution then the following set of statements must be true for any instance of a cphVB instruction.

The instruction has:

  * At least two operands
  * At most three operands
  * At most one destination operand
  * The first operand is always the destination operand

Whether an operand is an array, a scalar or a constant is determined at runtime by inspecting an instance of the struct:

.. doxygenstruct:: cphvb_instruction
   :project: cphVB
   :path: doxygen/xml

If operand[i] == NULL then the operand i is a constant and the value cphvb_instruction.constant is type cast to the appropriate type according to cphvb_instruction.constant_type and used as operand.

TODO: described instruction interpretation/handling, stuff like what is supposed to happen to inst.status.

Data Structures and Types
~~~~~~~~~~~~~~~~~~~~~~~~~

Scalars, constants and array elements all belong to one of the basic types defined cphvb_type.h.


Component Communication
~~~~~~~~~~~~~~~~~~~~~~~

...

Component Configuration
~~~~~~~~~~~~~~~~~~~~~~~

...


Core
----

Bridges / Language frontends
----------------------------

...

NumPy: Python
~~~~~~~~~~~~~

...

Microsoft CIL: C# / F# / VB.NET
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

...

NDArray: C++
~~~~~~~~~~~~

...

Vector Engine Managers
----------------------

...

node
~~~~

...

cluster
~~~~~~~

...

Vector Engines
--------------

...


score
~~~~~

...

mcore
~~~~~

...

gpu
~~~

...

