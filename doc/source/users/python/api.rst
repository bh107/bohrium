Python API
~~~~~~~~~~

Bohrium inherit must of the NumPy API. It is not the whole of the NumPy API that is accelerate but they are all usable and can be found under the same module and name as in NumPy. The following is the part of the Bohrium API that is accelerated and specialized to Bohrium.

Module contents
---------------

.. automodule:: bohrium
    :members: flush

Bohrium's ndarray
-----------------

.. autoclass:: bohrium._bh.ndarray

bohrium.array\_create module
----------------------------

.. automodule:: bohrium.array_create
    :members:

bohrium.backend\_messaging module
---------------------------------

.. automodule:: bohrium.backend_messaging
    :members:
    :show-inheritance:

bohrium.bhary module
--------------------

.. automodule:: bohrium.bhary
    :members:
    :show-inheritance:

bohrium.blas module
-------------------

.. automodule:: bohrium.blas
    :members:
    :show-inheritance:

bohrium.concatenate module
--------------------------

.. automodule:: bohrium.concatenate
    :members:
    :show-inheritance:

bohrium.contexts module
-----------------------

.. automodule:: bohrium.contexts
    :members:
    :show-inheritance:

bohrium.interop\_numpy module
-----------------------------

.. automodule:: bohrium.interop_numpy
    :members:
    :show-inheritance:

bohrium.interop\_pycuda module
------------------------------

.. automodule:: bohrium.interop_pycuda
    :members:
    :show-inheritance:

bohrium.interop\_pyopencl module
--------------------------------

.. automodule:: bohrium.interop_pyopencl
    :members:
    :show-inheritance:

bohrium.linalg module
---------------------

.. automodule:: bohrium.linalg
    :members: gauss, lu, solve, jacobi, matmul, dot, norm, tensordot, solve_tridiagonal, cg

bohrium.loop module
-------------------

.. automodule:: bohrium.loop
    :members:
    :show-inheritance:

bohrium.random123 module
------------------------

.. automodule:: bohrium.random123
    :members: seed, random_sample, randin, uniform, rand, randn, random_integers, standard_normal, normal, standard_exponential, exponential

bohrium.signal module
---------------------

.. automodule:: bohrium.signal
    :members: correlate1d, convolve1d, convolve, correlate

bohrium.summations module
-------------------------

.. automodule:: bohrium.summations
    :members:
    :show-inheritance:

bohrium.user\_kernel module
---------------------------

.. automodule:: bohrium.user_kernel
    :members:
    :show-inheritance:


