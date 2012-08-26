.. cphVB documentation master file

Welcome!
=================================

cphVB provides a runtime environment for efficiently executing vectorized applications using your favorourite programming languange Python/NumPy, C#, F# on Linux, Windows and MacOSX.

Forget handcrafting CUDA/OpenCL to utilize your GPU, forget threading, mutexes and locks to utilize your multi-core CPU and forget about MPI to program your cluster just cphVB!

Features
--------

+-----------+----------------+---------------+-------+----+----+
|           | Architecture Support           | Frontends       |
+-----------+----------------+---------------+-------+----+----+
|           | Multi-Core CPU | Many-Core GPU | NumPy | F# | C# |
+===========+================+===============+=======+====+====+
| Linux     | X              | X             | X     | x  | x  |
+-----------+----------------+---------------+-------+----+----+
| MacOSX    | -              | -             | X     | x  | x  |
+-----------+----------------+---------------+-------+----+----+
| Windows   | -              | X             | -     | x  | x  |
+-----------+----------------+---------------+-------+----+----+

Get Started!
------------

* `Downloads <https://bitbucket.org/cphvb/cphvb/downloads/>`_ 

.. toctree::
   :maxdepth: 2

   installation/index
   users/index

.. toctree::
   :maxdepth: 1

   developers/index

.. toctree::
   :maxdepth: 1

   faq
   bugs
   license

