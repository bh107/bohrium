.. cphVB documentation master file

Welcome!
=================================

cphVB provides a runtime environment for efficiently executing vectorized applications using your favorourite programming languange Python/NumPy, C#, F#, C, C++ on  Linux, Windows and MacOSX.

Forget handcrafting CUDA/OpenCL to utilize your GPU, forget threading, mutexes and locks to utilize your multi-core CPU and forget about MPI to program your cluster just cphVB!

Features
--------

+-----------+----------------+---------------+---------+-------+-----+----+----+
|           | Architecture Support                     | Frontends             |
+-----------+----------------+---------------+---------+-------+-----+----+----+
|           | Multi-Core CPU | Many-Core GPU | Cluster | NumPy | C++ | F# | C# |
+===========+================+===============+=========+=======+=====+====+====+
| Linux     | X              | X             | -       | X     | x   | x  | x  |
+-----------+----------------+---------------+---------+-------+-----+----+----+
| MacOSX    | X              | -             | -       | -     | x   | x  | x  |
+-----------+----------------+---------------+---------+-------+-----+----+----+
| Windows   | X              | X             | -       | -     | x   | x  | x  |
+-----------+----------------+---------------+---------+-------+-----+----+----+

Get Started!
------------

.. toctree::
   :maxdepth: 2

   installation/index
   users/index
   developers/guide
   developers/tools

.. toctree::
   :maxdepth: 1

   faq
   bugs
   license

