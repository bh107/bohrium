Python/NumPy
------------

Bohrium implements a new python module ``bohrium`` that introduces a new array class ``bohrium.ndarray`` which inherent from ``numpy.ndarray``. The two array classes a fully compatible thus one only has to replace ``numpy.ndarray`` with ``bohrium.ndarray`` in order to utilize the Bohrium runtime system.

The following example is a heat-equation solver that uses Bohrium. Note that the only different between Bohrium code and NumPy code is the first line where we import bohrium as np instead of numpy as np::

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

    python -m bohrium heat2d.py

In this case, all instances of ``import numpy`` is converted to ``import bohrium`` seamlessly. If you need to access the real numpy module use ``import numpy_force``.


Acceleration
~~~~~~~~~~~~

The approach of Bohrium is to accelerate all element-wise functions in NumPy (aka universals functions) as well as the reductions and accumulations of element-wise functions. This approach makes it possible to accelerate the heat-equation solver on both multi-core CPUs and GPUs.

Beside element-wise functions, Bohrium also accelerate a selection of common NumPy functions such as ``dot()`` and ``solve()``. But the number of functions in NumPy and related projects such as SciPy is enormous thus we cannot hope to accelerate every single function in Bohrium. Instead, Bohrium will automatically convert ``bohrium.ndarray`` to ``numpy.ndarray`` when encountering a function that Bohrium cannot accelerate. When running on the CPU, this conversion is very cheap but when running on the GPU, this conversion requires the array data to be copied from the GPU to the CPU.

Matplotlib’s ``matshow()`` function is example of a function Bohrium cannot accelerate. Say we want to visualize the result of the heat-equation solver, we could use ``matshow()``::

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to convert between Bohrium and NumPy explicitly and thus avoid Python warnings. Let's walk through an example:

Create a new NumPy array with ones::

    np_ary = numpy.ones(42)

Convert any type of array to Bohrium::

    bh_ary = bohrium.array(np_ary)

Copy a bohrium array into a new NumPy array::

    npy2 = bh_ary.copy2numpy()


    python -c "import bohrium as bh; print(bh.bh_info.runtime_info())"
