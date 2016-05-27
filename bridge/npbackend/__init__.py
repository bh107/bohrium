"""
The module initialization of npbackend/bohrium imports and exposes all methods
required to become a drop-in replacement for numpy.
"""
import sys
if 'numpy_force' not in sys.modules:
    import numpy
    sys.modules['numpy_force'] = numpy
    del numpy

#We import all of NumPy and overwrite with the objects we implement ourself
from numpy_force import *

from .array_create import *
from .array_manipulation import *
from .ufunc import UFUNCS, gather
from .bhary import check, check_biclass, fix_biclass, in_bhmem
from ._info import numpy_types
from ._util import flush
from . import linalg
from .linalg import matmul, dot, tensordot
from .summations import sum, prod, max, min
from numpy_force import dtype
asarray = array
asanyarray = array

# Expose all ufuncs
for f in UFUNCS:
    exec("%s = f" % f.info['name'])

# Aliases
aliases = [
    ('abs', 'absolute')
]

for f, t in aliases:
    exec("%s = %s" % (f, t))

# Expose all data types
for t in numpy_types:
    exec("%s = numpy.%s" % (t.__str__(), t.__str__()))

# Type aliases
type_aliases = [
    ('bool',    'bool'),
    ('int',     'int'),
    ('uint',    'numpy.uint64'),
    ('float',   'float'),
    ('complex', 'complex')
]

for f, t in type_aliases:
    exec("%s = %s" % (f, t))

# Note that the following modules needs ufuncs and dtypes
from . import random123 as random

# Some modules (e.g. scipy) accesses '__all__' directly
__all__ = [x for x in dir() if not x.startswith("_")]

#Finally, let's bohriumify the exposed API
import bohriumify
bohriumify.modules()
