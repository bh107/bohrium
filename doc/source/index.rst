.. cphVB documentation master file

Welcome!
========

cphVB provides a runtime environment for efficiently executing vectorized applications using your favorourite programming languange Python/NumPy, C#, F# on Linux, Windows and MacOSX.

Forget handcrafting CUDA/OpenCL to utilize your GPU, forget threading, mutexes and locks to utilize your multi-core CPU and forget about MPI to program your cluster just cphVB!

Features
--------

+-----------+-----------------+----------------+---------------+-------+----+----+
|           | Architecture Support             | Frontends                       |
+-----------+-----------------+----------------+---------------+-------+----+----+
|           | Single-Core CPU | Multi-Core CPU | Many-Core GPU | NumPy | F# | C# |
+===========+=================+================+===============+=======+====+====+
| Linux     | X               | X              | X             | X     | X  | X  |
+-----------+-----------------+----------------+---------------+-------+----+----+
| MacOSX    | X               | -              | -             | X     | X  | X  |
+-----------+-----------------+----------------+---------------+-------+----+----+
| Windows   | X               | X              | -             | -     | X  | X  |
+-----------+-----------------+----------------+---------------+-------+----+----+

Get Started!
------------

* `Downloads <https://bitbucket.org/cphvb/cphvb/downloads/>`_ 

.. toctree::
   :maxdepth: 2

   installation/index

.. toctree::
   :maxdepth: 2

   users/index

.. toctree::
   :maxdepth: 1

   developers/index

.. toctree::
   :maxdepth: 1

   faq
   bugs
   license

