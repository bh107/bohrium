=======================================
Status of this collection of Benchmarks
=======================================

MacOSX 10.9.4, NumPy 1.6.2 and 1.8.1
Ubuntu 12.04.4, NumPy 1.8.1

LMM_swaption_vec.py::

  OSX
  + Runs with NumPy
  - Fails with Bohrium:
    AttributeError: 'module' object has no attribute 'concatenate'

  Ubuntu
  + Runs with NumPy

black_scholes.py::

  + No failures observed.

convolve.py::

  OSX
  - Fails with NumPy
    TypeError: unsupported operand type(s) for *: 'instance' and 'float'
  - Fails with Bohrium
    Crash and burn.

  Ubuntu
  - Fails with NumPy
    benchmark/Python/convolve.py:19: RuntimeWarning: Encountering an operation
    not supported by Bohrium. It will be handled by the original NumPy.
        totalsum += kernel[filterY + kernelrad, filterX + kernelrad]

convolve_2d.py::

  OsX
  - Fails with NumPy:
    SystemError: error return without exception set
  - Fails with Bohrium:
    SystemError: error return without exception set

  Ubuntu
  - Fails with NumPy:

    Traceback (most recent call last):
        File "benchmark/Python/convolve_2d.py", line 51, in <module>
          image, image_filter = convolve_2d_init(N)
        File "benchmark/Python/convolve_2d.py", line 34, in convolve_2d_init
          kernel  = gen_2d_filter(fsize, 13.0)
        File "benchmark/Python/convolve_2d.py", line 21, in gen_2d_filter
          kernel[filterY + kernelrad,filterX + kernelrad] = caleuler * np.exp(-distance) 
      IndexError: index 10 is out of bounds for axis 1 with size 10

convolve_3d.py::

  OSX
  - Fails with NumPy:
    TypeError: random_sample() got an unexpected keyword argument 'dtype'
  - Fails with Bohrium:
    Crash and burn.

  Ubuntu
  - Fails with NumPy:
    Traceback (most recent call last):
      File "benchmark/Python/convolve_3d.py", line 49, in <module>
        image, image_filter = convolve_3d_init(N)
      File "benchmark/Python/convolve_3d.py", line 31, in convolve_3d_init
        rgb     = np.random.random((512, 512, 512), dtype=datatype)
      File "mtrand.pyx", line 730, in mtrand.RandomState.random_sample (numpy/random/mtrand/mtrand.c:6645)
    TypeError: random_sample() got an unexpected keyword argument 'dtype'

convolve_seperate_std.py::

  OSX
  + Runs with NumPy
  + Runs with Bohrium

  Ubuntu
  + Runs with NumPy

gauss.py::

  + No failures observed.

heat_equation.py::
  
  + No failures observed.

jacobi.py::

  - Does not seem to converge... ever...

jacobi_fixed.py::

  OSX
  + Runs with NumPy
  + Runs with Bohrium

  Ubuntu

jacobi_stencil.py::

  OSX
  + Runs with NumPy 1.6.2 + 1.8.1
  + Runs with Bohrium

  Ubuntu
  + RUns with NumPy

knn.naive.py::

  OSX
  + Runs with NumPy 1.6.2 + 1.8.1
  + Runs with Bohrium

  Ubuntu
  + RUns with NumPy

knn.py::

  OSX
  + Runs with NumPy 1.6.2 + 1.8.1
  - Fails with Bohrium
    AttributeError: 'module' object has no attribute 'max'

  Ubuntu
  + RUns with NumPy

lattice_boltzmann_D2Q9.py::

  OSX
  + Runs with NumPy 1.6.2 + 1.8.1
  - Fails with Bohrium due to missing 'asarray'

  Ubuntu
  + RUns with NumPy

lbm.3d.py::
  
  OSX
  + Runs with NumPy 1.6.2 + 1.8.1
  + Runs with Bohrium

  Ubuntu
  + Runs with NUmPy

lu.py::
  
  OSX
  + Runs with NumPy 1.6.2 + 1.8.1
  + Runs with Bohrium

  Ubuntu
  + Runs with NUmPy

mc.py::

  OSX
  - Fails with NumPy 1.6.2 + 1.8.1:
    TypeError: random_sample() got an unexpected keyword argument 'dtype'
  + Runs with Bohrium

  Ubuntu
  + Runs with NUmPy

mxmul.py::

  + No failures observed.

nbody.py::

  + No failures observed.

ndstencil.py::

  OSX
  - Rails with NumPy 1.6.2 + 1.8.1:
    TypeError: random_sample() got an unexpected keyword argument 'dtype'
  + Runs with Bohrium

  Ubuntu

point27.py::

  OSX
  + Runs with NumPy 1.6.2 + 1.8.1
  + Runs with Bohrium

  Ubuntu

shallow_water.py::

  OSX
  + Runs with NumPy 1.6.2 + 1.8.1
  + Runs with Bohrium

  Ubuntu

sor.py::

  OSX
  + Runs with NumPy 1.6.2 + 1.8.1
  + Runs with Bohrium

  Ubuntu

synth.py::

  OSX
  + Runs with NumPy 1.6.2 + 1.8.1
  + Runs with Bohrium

  Ubuntu

wireworld.py::

  OSX
  + Runs with NumPy 1.6.2 + 1.8.1
  - Does not run with Bohrium due to missing '.tile'

  Ubuntu

