.. Bohrium documentation master file

Welcome!
========

Bohrium provides a runtime environment for efficiently executing vectorized applications using your favorourite programming languange Python/NumPy, C#, F# on Linux, Windows and MacOSX.

Forget handcrafting CUDA/OpenCL to utilize your GPU, forget threading, mutexes and locks to utilize your multi-core CPU and forget about MPI to program your cluster just Bohrium!

Features
--------

+-----------+-----+-----+-------------+-------+-----+------+
|           | Architecture Support    | Frontends          |
+-----------+-----+-----+-------------+-------+-----+------+
|           | CPU | GPU | CPU Cluster | NumPy | C++ | .NET |
+===========+=====+=====+=============+=======+=====+======+
| Linux     | X   | X   | X           | X     | X   |  X   |
+-----------+-----+-----+-------------+-------+-----+------+
| MacOSX    | X   | X   |             | X     |     |  X   |
+-----------+-----+-----+-------------+-------+-----+------+
| Windows   | X   |     |             |       |     |  X   |
+-----------+-----+-----+-------------+-------+-----+------+

Get Started!
------------

.. * `Downloads <https://bitbucket.org/bohrium/bohrium/downloads/>`_

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

