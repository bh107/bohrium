"""
CphVB NumPy is an extended version of NumPy that supports cphVB as the computation backend.  The cphVB NumPy module includes the original NumPy backend, which makes it possible to use both backend through the same Python Module. 

In order to use the cphVB backend rather than normal NumPy you can either import the ``cphvbnumpy`` module instead of ``numpy`` or use the new optional parameter `cphvb`.

The easiest method it simple to change the import statement as illustrated in the following two code examples. 

A regular NumPy execution::   
 
  >>> import numpy as np
  >>> A = np.ones((3,))
  >>> B = A + 42
  >>> print B
  [ 43 43 43 ]

A cphVB execution::

  >>> import cphvbnumpy as np
  >>> A = np.ones((3,))
  >>> B = A + 42
  >>> print B
  [ 43 43 43 ]

Alternatively, most array creation methods supports the optional parameter ``cphvb``::

  >>> import numpy as np
  >>> A = np.ones((3,), cphVB=True)
  >>> B = A + 42
  >>> print B
  [ 43 43 43 ]

Backend Transition
~~~~~~~~~~~~~~~~~~

It is possible to change the backend of an existing array by accessing the ``.cphvb`` attribute::

  >>> import cphvbnumpy as np
  >>> A = np.ones((3,), cphVB=True)
  >>> print A.cphvb
  True
  >>> A.cphvb = False
  >>> print A.cphvb
  False

All arrays will use the cphVB backend when combining arrays that use different backends. The following code example will be computed by the cphVB backend::
    
  >>> import cphvbnumpy as np
  >>> A = np.ones((3,), cphVB=True)
  >>> B = np.ones((3,), cphVB=False)
  >>> C = A + B
  >>> print C.cphvb
  True

The cphVB backend does not support all the functionality in NumPy. Therefore, some functions will always use the NumPy backend even though the user specified the cphVB backend. In such cases, the execution will raise a Python warning. 


Duality of cphvbnumpy and numpy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``cphvbnumpy`` module is an extension to the ``numpy`` module. Therefore, when importing cphvbnumpy the ``numpy`` module is also imported. The ``cphvbnumpy`` module introduces some new functions and overwrites some existing ``numpy`` functions. However, functions not implemented in the cphvbnumpy module will automatically use the ``numpy`` module.

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
    Code-examples of using Python/NumPy with CphVB.

"""
from core import *
import linalg

from numpy import bool8 as bool
from numpy import amin as min
from numpy import amax as max

