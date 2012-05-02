cphVB Developer Guide
=======================

...

Tools and Environment
---------------------

Tools of the trade::

  sudo apt-get install git valgrind 

Valgrind and Python
~~~~~~~~~~~~~~~~~~~

Valgrind is a great tool for memory debugging, memory leak detection, and profiling.
However, both Python and NumPy floods the valgrind output with memory errors - it is therefore necessary to use a debug and valgrind friendly version of Python::

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

Building and Installing
-----------------------

This requires installing some dependencies in addition to the tools above::

  sudo apt-get install python-dev mpi-default-dev

Get the source-code::

  git clone git@bitbucket.org:cphvb/cphvb-priv.git
  cd cphvb-priv
  git submodule init
  git submodule update

Before compiling remember to set the $PYTHON environment variable. This is useful if you wish to debug the Python bridge with valgrind. Such as::

  PYTHON=dython

Then compile and install it::

  ./build install

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

