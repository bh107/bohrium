.. Bohrium documentation master file

Welcome!
========

Bohrium provides a runtime environment for efficiently executing vectorized applications using your favorite programming language Python/NumPy, C#, F# on Linux, Windows and MacOSX.

Forget handcrafting CUDA/OpenCL to utilize your GPU, forget threading, mutexes and locks to utilize your multi-core CPU and forget about MPI to program your cluster just Bohrium!

Features
--------

+-----------+-----------------+-----------------+---------------+---------------+-----+------+
|           |     Architecture Support          |           Frontends                        |
+-----------+-----------------+-----------------+---------------+---------------+-----+------+
|           | Multi-Core CPU  | Many-Core GPU   | Python2/NumPy | Python3/NumPy | C++ | .NET |
+===========+=================+=================+===============+===============+=====+======+
| Linux     | X               |  X              | X             | X             | X   |  X   |
+-----------+-----------------+-----------------+---------------+---------------+-----+------+
| MacOSX    | X               |                 | X             |               |     |  X   |
+-----------+-----------------+-----------------+---------------+---------------+-----+------+
| Windows   |                 |                 |               |               |     |      |
+-----------+-----------------+-----------------+---------------+---------------+-----+------+

- **Lazy Evaluation**, Bohrium will lazy evaluate all Python/NumPy operations until it encounters a “Python Read” such a printing an array or having a if-statement testing the value of an array.
- **Views** Bohrium supports NumPy views fully thus operating on array slices does not involve data copying. 
- **Loop Fusion**, Bohrium uses a `fusion algorithm <http://dl.acm.org/citation.cfm?id=2967945>`_ that fuses (or merges) array operations into the same computation kernel that are then JIT-compiled and executed. However, Bohrium can only fuse operations that have some common sized dimension and no horizontal data conflicts. 
- **Lazy CPU/GPU Communiction**, Bohrium only move data between the host and the GPU when the data is accessed directly by Python or a Python C-extension.


Please note:
    * Bohrium is a 64bit project exclusively.
    * We are working on a Windows version that uses the .NET frontend and CPU backend.

.. raw:: html

   <script>
     ((window.gitter = {}).chat = {}).options = {
       room: 'bh107/Lobby'
     };
   </script>
   <script src="https://sidecar.gitter.im/dist/sidecar.v1.js" async defer></script>

Get Started!
------------

.. toctree::
   :maxdepth: 2

   installation/index

.. toctree::
   :maxdepth: 2

   users/index

.. toctree::
   :maxdepth: 1

   developers/index

* `Student Projects <https://docs.google.com/document/d/1bMxKi86q-F0kVHLZoQzzt7BhTde4GOco9cA_fhJf45c/edit?usp=sharing/>`_

* `Benchmark Suite <http://benchpress.readthedocs.org/>`_

.. toctree::
   :maxdepth: 1

   faq
   bugs
   publications
   contact
   license

