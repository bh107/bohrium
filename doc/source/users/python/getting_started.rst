Getting Started
~~~~~~~~~~~~~~~

Bohrium implements a new python module ``bohrium`` that introduces a new array class ``bohrium.ndarray`` which inherits from ``numpy.ndarray``. The two array classes are fully compatible thus you only has to replace ``numpy.ndarray`` with ``bohrium.ndarray`` in order to utilize the Bohrium runtime system. Alternatively, in order to have Bohrium replacing NumPy automatically, you  can use the ``-m bohrium`` argument when running Python::

    $ python -m bohrium my_numpy_app.py

In order to choose which Bohrium backend to use, you can define the ``BH_STACK`` environment variable. Currently, three backends exist: ``openmp``, ``opencl``, and ``cuda``.

Before using Bohrium, you can check the current runtime configuration using::

    $ BH_STACK=opencl python -m bohrium --info

    ----
    Bohrium version: 0.10.2.post8
    ----
    Bohrium API version: 0.10.2.post8
    Installed through PyPI: False
    Config file: ~/.bohrium/config.ini
    Header dir: ~/.local/lib/python3.7/site-packages/bohrium_api/include
    Backend stack:
    ----
    OpenCL:
      Device[0]: AMD Accelerated Parallel Processing / Intel(R) Core(TM) i7-5600U CPU @ 2.60GHz (OpenCL C 1.2 )
      Memory:         7676 MB
      Malloc cache limit: 767 MB (90%)
      Cache dir: "~/.local/var/bohrium/cache"
      Temp dir: "/tmp/bh_75cf_314f5"
      Codegen flags:
        Index-as-var: true
        Strides-as-var: true
        const-as-var: true
    ----
    OpenMP:
      Main memory: 7676 MB
      Hardware threads: 4
      Malloc cache limit: 2190 MB (80% of unused memory)
      Cache dir: "~/.local/var/bohrium/cache"
      Temp dir: "/tmp/bh_75a5_c6368"
      Codegen flags:
        OpenMP: true
        OpenMP+SIMD: true
        Index-as-var: true
        Strides-as-var: true
        Const-as-var: true
      JIT Command: "/usr/bin/cc -x c -fPIC -shared  -std=gnu99  -O3 -march=native -Werror -fopenmp -fopenmp-simd -I~/.local/share/bohrium/include {IN} -o {OUT}"
    ----

Notice, since ``BH_STACK=opencl`` is defined, the runtime stack consist of both the OpenCL and the OpenMP backend. In this case, OpenMP only handles operations unsupported by OpenCL.


Heat Equation Example
---------------------

The following example is a heat-equation solver that uses Bohrium. Note that the only difference between Bohrium code and NumPy code is the first line where we import bohrium as np instead of numpy as np::

    import bohrium as np
    def heat2d(height, width, epsilon=42):
      G = np.zeros((height+2,width+2),dtype=np.float64)
      G[:,0]  = -273.15
      G[:,-1] = -273.15
      G[-1,:] = -273.15
      G[0,:]  = 40.0
      center = G[1:-1,1:-1]
      north  = G[:-2,1:-1]
      south  = G[2:,1:-1]
      east   = G[1:-1,:-2]
      west   = G[1:-1,2:]
      delta  = epsilon+1
      while delta > epsilon:
        tmp = 0.2*(center+north+south+east+west)
        delta = np.sum(np.abs(tmp-center))
        center[:] = tmp
      return center
    heat2d(100, 100)

Alternatively, you can import Bohrium as NumPy through the command line argument ``-m bohrium``::

    $ python -m bohrium heat2d.py

In this case, all instances of ``import numpy`` is converted to ``import bohrium`` seamlessly. If you need to access the real numpy module use ``import numpy_force``.


Acceleration
------------

The approach of Bohrium is to accelerate all element-wise functions in NumPy (aka universal functions) as well as the reductions and accumulations of element-wise functions. This approach makes it possible to accelerate the heat-equation solver on both multi-core CPUs and GPUs.

Beside element-wise functions, Bohrium also accelerates a selection of common NumPy functions such as ``dot()`` and ``solve()``. But the number of functions in NumPy and related projects such as SciPy is enormous thus we cannot hope to accelerate every single function in Bohrium. Instead, Bohrium will automatically convert ``bohrium.ndarray`` to ``numpy.ndarray`` when encountering a function that Bohrium cannot accelerate. When running on the CPU, this conversion is very cheap but when running on the GPU, this conversion requires the array data to be copied from the GPU to the CPU.

Matplotlib's ``matshow()`` function is example of a function Bohrium cannot accelerate. Say we want to visualize the result of the heat-equation solver, we could use ``matshow()``::

    from matplotlib import pyplot as plt

    res = heat2d(100, 100)
    plt.matshow(res, cmap='hot')
    plt.show()

.. image:: gfx/heat2d.png
   :scale: 80 %
   :align: center

Beside producing the image (after approx. 1 min), the execution will raise a Python warning informing you that matplotlib function is handled like a regular NumPy::

    /usr/lib/python2.7/site-packages/matplotlib/cbook.py:1506: RuntimeWarning:
    Encountering an operation not supported by Bohrium. It will be handled by the original NumPy.
    x = np.array(x, subok=True, copy=copy)

.. note:: Increasing the problem size will improve the performance of Bohrium significantly!


Convert between Bohrium and NumPy
---------------------------------

It is possible to convert between Bohrium and NumPy explicitly and thus avoid Python warnings. Let's walk through an example:

Create a new NumPy array with ones::

    np_ary = numpy.ones(42)

Convert any type of array to Bohrium::

    bh_ary = bohrium.array(np_ary)

Copy a bohrium array into a new NumPy array::

    npy2 = bh_ary.copy2numpy()

