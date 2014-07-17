=======================================
Status of this collection of Benchmarks
=======================================

MacOSX 10.9.4, NumPy 1.6.2 and 1.8.1
Ubuntu 12.04.4, NumPy 1.8.1

LMM_swaption_vec.py::

  OSX
  - Fails with Bohrium:

    Traceback (most recent call last):
      File "benchmark/Python/LMM_swaption_vec.py", line 58, in <module>
        eps = np.concatenate((eps_tmp,-eps_tmp), axis = 1)
    AttributeError: 'module' object has no attribute 'concatenate'

black_scholes.py::

  + No failures observed.

convolve.py::

  OSX
  - Fails with NumPy:

    Traceback (most recent call last):
      File "benchmark/Python/convolve.py", line 54, in <module>
        image, image_filter = convolve_init(N)
      File "benchmark/Python/convolve.py", line 34, in convolve_init
        rgb     = np.add.reduce((rgb*tones[np.newaxis, np.newaxis, :]), axis=2)
    TypeError: unsupported operand type(s) for *: 'instance' and 'float'

  - Fails with Bohrium

    Crash and burn.

convolve_2d.py::

  OSX
  - Fails with NumPy:

    Traceback (most recent call last):
      File "benchmark/Python/convolve_2d.py", line 51, in <module>
        image, image_filter = convolve_2d_init(N)
      File "benchmark/Python/convolve_2d.py", line 31, in convolve_2d_init
        rgb     = np.array(img, dtype=datatype)
    SystemError: error return without exception set

  - Fails with Bohrium:

    Traceback (most recent call last):
      File "benchmark/Python/convolve_2d.py", line 51, in <module>
        image, image_filter = convolve_2d_init(N)
      File "benchmark/Python/convolve_2d.py", line 31, in convolve_2d_init
        rgb     = np.array(img, dtype=datatype)
      File "/Users/slund/.local/lib/python2.7/site-packages/bohrium/ndarray.py", line 60, in inner
        ret = func(*args, **kwargs)
      File "/Users/slund/.local/lib/python2.7/site-packages/bohrium/array_create.py", line 123, in array
        subok=subok, ndmin=ndmin)
    SystemError: error return without exception set

convolve_3d.py::

  OSX
  - Fails with NumPy:
    Traceback (most recent call last):
      File "benchmark/Python/convolve_3d.py", line 49, in <module>
        image, image_filter = convolve_3d_init(N)
      File "benchmark/Python/convolve_3d.py", line 32, in convolve_3d_init
        kernel  = gen_3d_filter(fsize, 13.0)
      File "benchmark/Python/convolve_3d.py", line 22, in gen_3d_filter
        kernel[filterZ + kernelrad, filterY + kernelrad,filterX + kernelrad] = caleuler * np.exp(-distance)
    IndexError: index 10 is out of bounds for axis 2 with size 10

  - Fails with Bohrium:
    Crash and burn.

convolve_seperate_std.py::

  + No failures observed.

gauss.py::

  + No failures observed.

heat_equation.py::
  
  + No failures observed.

jacobi.py::

  OSX

  - Weird with NumPy
    - Does not seem to converge... ever...

  - Fails with Bohrium:
   - RuntimeError: The Array Data Protection could not mummap the data region: 0x7fd9a8e0e4f0 (size: 0). Returned error code by mmap: Invalid argument.

jacobi_fixed.py::

  + No failures observed.

jacobi_stencil.py::

  + No failures observed.

knn.naive.py::

  + No failures observed.

knn.py::

  - Fails with Bohrium

    Traceback (most recent call last):
      File "benchmark/Python/knn.py", line 32, in <module>
        main()
      File "benchmark/Python/knn.py", line 27, in main
        compute_targets(base, targets)
      File "benchmark/Python/knn.py", line 12, in compute_targets
        tmp = np.max(tmp, 0)
    AttributeError: 'module' object has no attribute 'max'

lattice_boltzmann_D2Q9.py::

  - Fails with Bohrium:

    Traceback (most recent call last):
      File "benchmark/Python/lattice_boltzmann_D2Q9.py", line 197, in <module>
        cylinder = cylinder(H, W, obstacle=False)
      File "benchmark/Python/lattice_boltzmann_D2Q9.py", line 42, in cylinder
        t_3d    = np.asarray(t)[:, np.newaxis, np.newaxis]
    AttributeError: 'module' object has no attribute 'asarray'

lbm.3d.py::
  
  + No failures observed.

lu.py::
  
  + No failures observed.

mc.py::

  + No failures observed.

mxmul.py::

  + No failures observed.

nbody.py::

  + No failures observed.

ndstencil.py::

  + No failures observed.

point27.py::

  + No failures observed.

shallow_water.py::

  + No failures observed.

sor.py::

  + No failures observed.

synth.py::

  + No failures observed.

wireworld.py::

  - Fails with Bohrium:

    Traceback (most recent call last):
      File "benchmark/Python/wireworld.py", line 55, in <module>
        world = wireworld_init(N)
      File "benchmark/Python/wireworld.py", line 15, in wireworld_init
        data[1:-1,1:-1] = np.tile(np.array([
    AttributeError: 'module' object has no attribute 'tile'

