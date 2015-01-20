"""
The module initialization of npbackend/bohrium imports and exposes all methods
required to become a drop-in replacement for numpy.
"""
import sys
if 'numpy_force' not in sys.modules:
    import numpy
    sys.modules['numpy_force'] = numpy
    del numpy

from .array_create import *
from .array_manipulation import *
from .ufunc import UFUNCS
from .ndarray import check, check_biclass, fix_biclass
from ._info import numpy_types
from ._util import flush
from . import linalg
from .linalg import matmul, dot
from .summations import sum, prod, max, min
from . import import_external
from numpy_force import dtype
asarray = array

# Expose all ufuncs
for f in UFUNCS:
    exec("%s = f" % f.info['name'])

# Expose all data types
for t in numpy_types:
    exec("%s = numpy.%s"%(t.__str__(),t.__str__()))

# Note that the following modules needs ufuncs and dtypes
from . import random123 as random

# TODO: import all numpy functions
from numpy import meshgrid

# Finally, we import and expose external libraries
numpy_interface = [
    "numpy.lib.stride_tricks.as_strided",
    "numpy.newaxis",
    "numpy.pi",
    "numpy.transpose"
]

for i in import_external.api(numpy_interface):
    exec(i)
