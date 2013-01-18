"""
Bohrium NumPy is an extended version of NumPy that supports Bohrium as the computation backend.  The Bohrium NumPy module includes the original NumPy backend, which makes it possible to use both backend through the same Python Module. 

In order to use the Bohrium backend rather than normal NumPy you can either import the ``bohrium`` module instead of ``numpy`` or use the new optional parameter `bohrium`.

The easiest method it simple to change the import statement as illustrated in the following two code examples. 

A regular NumPy execution::   
 
  >>> import numpy as np
  >>> A = np.ones((3,))
  >>> B = A + 42
  >>> print B
  [ 43 43 43 ]

A Bohrium execution::

  >>> import bohrium as np
  >>> A = np.ones((3,))
  >>> B = A + 42
  >>> print B
  [ 43 43 43 ]

Alternatively, most array creation methods supports the optional parameter ``bohrium``::

  >>> import numpy as np
  >>> A = np.ones((3,), bohrium=True)
  >>> B = A + 42
  >>> print B
  [ 43 43 43 ]

Backend Transition
~~~~~~~~~~~~~~~~~~

It is possible to change the backend of an existing array by accessing the ``.bohrium`` attribute::

  >>> import bohrium as np
  >>> A = np.ones((3,), bohrium=True)
  >>> print A.bohrium
  True
  >>> A.bohrium = False
  >>> print A.bohrium
  False

All arrays will use the Bohrium backend when combining arrays that use different backends. The following code example will be computed by the Bohrium backend::
    
  >>> import bohrium as np
  >>> A = np.ones((3,), bohrium=True)
  >>> B = np.ones((3,), bohrium=False)
  >>> C = A + B
  >>> print C.bohrium
  True

The Bohrium backend does not support all the functionality in NumPy. Therefore, some functions will always use the NumPy backend even though the user specified the Bohrium backend. In such cases, the execution will raise a Python warning. 


Duality of bohrium and numpy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``bohrium`` module is an extension to the ``numpy`` module. Therefore, when importing bohrium the ``numpy`` module is also imported. The ``bohrium`` module introduces some new functions and overwrites some existing ``numpy`` functions. However, functions not implemented in the bohrium module will automatically use the ``numpy`` module.

.. note:: The NumPy functions not defined here will use the regular NumPy implementation. Hence, the documentation is available at the `Numpy Reference Guide <http://docs.scipy.org/doc/numpy/reference/>`_.

Available modules
~~~~~~~~~~~~~~~~~
core
   The essential functions, such as all the array creation functions.
 
linalg
    Common linear algebra functions

Available subpackages
~~~~~~~~~~~~~~~~~~~~~
examples
    Code-examples of using Python/NumPy with Bohrium.

"""
from core import *
import linalg

from numpy import bool8 as bool
from numpy import amin as min
from numpy import amax as max

