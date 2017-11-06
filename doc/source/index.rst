.. Bohrium documentation master file

Welcome!
========

Bohrium provides automatic acceleration of array operations in Python/NumPy, C, and C++ targeting multi-core CPUs and GP-GPUs.
Forget handcrafting CUDA/OpenCL to utilize your GPU and forget threading, mutexes and locks to utilize your multi-core CPU just use Bohrium!

Features
--------

+-----------+-----------------+-----------------+---------------+---------------+-----+------+
|           |     Architecture Support          |           Frontends                        |
+-----------+-----------------+-----------------+---------------+---------------+-----+------+
|           | Multi-Core CPU  | Many-Core GPU   | Python2/NumPy | Python3/NumPy | C   | C++  |
+===========+=================+=================+===============+===============+=====+======+
| Linux     | ✓               |  ✓              | ✓             | ✓             | ✓   |  ✓   |
+-----------+-----------------+-----------------+---------------+---------------+-----+------+
| Mac OS    | ✓               |  ✓              | ✓             |               | ✓   |  ✓   |
+-----------+-----------------+-----------------+---------------+---------------+-----+------+
| Windows   |                 |                 |               |               |     |      |
+-----------+-----------------+-----------------+---------------+---------------+-----+------+

- **Lazy Evaluation**, Bohrium will lazy evaluate all Python/NumPy operations until it encounters a “Python Read” such a printing an array or having a if-statement testing the value of an array.
- **Views** Bohrium supports NumPy views fully thus operating on array slices does not involve data copying.
- **Loop Fusion**, Bohrium uses a `fusion algorithm <http://dl.acm.org/citation.cfm?id=2967945>`_ that fuses (or merges) array operations into the same computation kernel that are then JIT-compiled and executed. However, Bohrium can only fuse operations that have some common sized dimension and no horizontal data conflicts.
- **Lazy CPU/GPU Communication**, Bohrium only moves data between the host and the GPU when the data is accessed directly by Python or a Python C-extension.
- **python -m bohrium**, automatically makes ``import numpy`` use Bohrium.
- `Jupyter Support <http://jupyter.org/>`_, you can use the magic command ``%%bohrium`` to automatically use Bohrium as NumPy.
- **Zero-copy** :ref:`interop` **with:**
    - `NumPy <http://www.numpy.org/>`_
    - `Cython <http://cython.org/>`_
    - `PyOpenCL <https://mathema.tician.de/software/pyopencl/>`_
    - `PyCUDA <https://mathema.tician.de/software/pycuda/>`_


Please note:
    * Bohrium is a 64-bit project exclusively.
    * We are working on a Windows version.
    * Source code is available here: https://github.com/bh107/bohrium

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
   license

