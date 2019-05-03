Getting Started
~~~~~~~~~~~~~~~

Bh107 implements a new python module ``bh107`` that introduces a new array class :func:`bh107.BhArray` which imitation :func:`numpy.ndarray`. The two array classes are zero-copy compatible thus you can convert a  :func:`bh107.BhArray` to a :func:`numpy.ndarray` without any data copy.

In order to choose which Bohrium backend to use, you can define the ``BH_STACK`` environment variable. Currently, three backends exist: ``openmp``, ``opencl``, and ``cuda``.

Before using Bohrium, you can check the current runtime configuration using::

    $ BH_STACK=opencl python -m bohrium_api --info

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

The following example is a heat-equation solver that uses Bh107. Note that the only difference between Bohrium code and NumPy code is the first line where we import bohrium as np instead of numpy as np::

    import bh107 as np
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
        delta = np.add.reduce(np.abs(tmp-center))
        center[:] = tmp
      return center
    heat2d(100, 100)


Convert between Bh107 and NumPy
-------------------------------

Create a new NumPy array with ones::

    np_ary = numpy.ones(42)

Convert any type of array to Bh107::

    bh_ary = bh107.array(np_ary)

Copy a Bh107 array into a new NumPy array::

    npy2 = bh_ary.copy2numpy()

Zero-copy a Bh107 array into a NumPy array::

    npy3 = bh_ary.asnumpy()
    # At this point `bh_ary` and `npy3` points to the same data.

