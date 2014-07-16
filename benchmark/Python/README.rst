=======================================
Status of this collection of Benchmarks
=======================================

LMM_swaption_vec.py::

  + Runs with NumPy
  - Fails with Bohrium:
    AttributeError: 'module' object has no attribute 'concatenate'

black_scholes.py::

  - Fails with NumPy 1.6.2:
    TypeError: random_sample() got an unexpected keyword argument 'dtype'
  + Runs with Bohrium

convolve.py::

  - Fails with NumPy 1.6.2:
    TypeError: unsupported operand type(s) for *: 'instance' and 'float'
  - Fails with Bohrium:
    Crash and burn.

convolve_2d.py::

  - Fails with NumPy 1.6.2:
    SystemError: error return without exception set
  - Fails with Bohrium:
    SystemError: error return without exception set

convolve_3d.py::

  - Fails with NumPy 1.6.2:
    TypeError: random_sample() got an unexpected keyword argument 'dtype'
  - Fails with Bohrium:
    Crash and burn.

convolve_seperate_std.py::

  + Runs with NumPy 1.6.2
  + Runs with Bohrium

gauss.py::

  - Fails with NumPy 1.6.2
    TypeError: random_sample() got an unexpected keyword argument 'dtype'
  + Runs with Bohrium

heat_equation.py::
  
  + Runs with NumPy 1.6.2
  + Runs with Bohrium

jacobi.py::

  + Runs with NumPy 1.6.2
  - Fails with Bohrium::
    RuntimeError: The Array Data Protection could not mummap the data region:
    0x7fc1226625f0 (size: 0).Returned error code by mmap: Invalid argument.

jacobi_fixed.py::

  + Runs with NumPy 1.6.2
  + Runs with Bohrium

jacobi_stencil.py::

  + Runs with NumPy 1.6.2
  + Runs with Bohrium

knn.naive.py::

  + Runs with NumPy 1.6.2
  + Runs with Bohrium

knn.py::

  + Runs with NumPy 1.6.2
  - Fails with Bohrium
    AttributeError: 'module' object has no attribute 'max'

lattice_boltzmann_D2Q9.py::

  + Runs with NumPy 1.6.2
  - Fails with Bohrium due to missing 'asarray'

lbm.3d.py::
  
  + Runs with NumPy 1.6.2
  + Runs with Bohrium

lu.py::
  
  + Runs with NumPy 1.6.2
  + Runs with Bohrium

mc.py::

  - Fails with NumPy 1.6.2:
    TypeError: random_sample() got an unexpected keyword argument 'dtype'
  + Runs with Bohrium

mxmul.py::

  - Fails with NumPy 1.6.2:
    AttributeError: 'bohrium.ndarray' object has no attribute 'bohrium'
  - Fails with Bohrium, fix use of .bohrium

nbody.py::

  + Runs with NumPy 1.6.2
  + Runs with Bohrium

ndstencil.py::

  - Rails with NumPy 1.6.2:
    TypeError: random_sample() got an unexpected keyword argument 'dtype'
  + Runs with Bohrium

point27.py::

  + Runs with NumPy 1.6.2
  + Runs with Bohrium

shallow_water.py::

  + Runs with NumPy 1.6.2
  + Runs with Bohrium

sor.py::

  + Runs with NumPy 1.6.2
  + Runs with Bohrium

synth.py::

  + Runs with NumPy 1.6.2
  + Runs with Bohrium

wireworld.py::

  + Runs with NumPy 1.6.2
  - Does not run with Bohrium due to missing '.tile'
