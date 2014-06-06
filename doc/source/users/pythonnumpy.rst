Python/NumPy
------------

Bohrium implements a new python module `bohrium` that introduces a new array class `bohrium.ndarray` which inherent from `numpy.ndarray`. The two array classes a fully compatible thus one only has to replace `numpy.ndarray` with `bohrium.ndarray` in order to utilize the Bohrium runtime system.

The following example is a heat-equation solver that uses Bohrium. Note that the only different between Bohrium code and NumPy code is the first line where we import bohrium as np instead of numpy as np::

    import bohrium as np
    def solve(grid, iter):
        center = grid[1:-1,1:-1]
        north  = grid[ :-2,1:-1]
        south  = grid[2:  ,1:-1]
        east   = grid[1:-1,2:  ]
        west   = grid[1:-1, :-2]
        for _ in xrange(iter):
            tmp = 0.2*(center+north+south+east+west)
            delta = np.sum(np.absolute(tmp-center))
            center[:] = tmp
    grid = np.arange(width**2).reshape((100,100))
    solve(grid, 42)


Library Reference
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: pythonnumpygen

   bohrium.core
   bohrium.linalg
   bohrium.examples

Glossary
~~~~~~~~

* :ref:`genindex`
