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
from .backend_messaging import *
from .array_manipulation import *
from .reorganization import *
from .contexts import *
from .ufuncs import UFUNCS
from .masking import *
from .bhary import check, fix_biclass, in_bhmem
from ._info import numpy_types
from ._util import flush
from . import linalg
from .linalg import matmul, dot, tensordot
from .summations import *
from .disk_io import *
from . import contexts
from numpy_force import dtype
asarray = array
asanyarray = array


def replace_numpy(function):
    def wrapper(*args, **kwargs):
        with contexts.BohriumContext():
            # Run your function/program
            result = function(*args, **kwargs)
        return result
    return wrapper


# Expose all ufuncs
for _name, _f in UFUNCS.items():
    exec("%s = _f" % _name)

# Aliases
_aliases = [
    ('abs', 'absolute'),
    ('round', 'round_')
]
for _f, _t in _aliases:
    exec("%s = %s" % (_f, _t))
cumsum = add.accumulate
cumprod = multiply.accumulate

# Expose all data types
for _t in numpy_types:
    exec("%s = numpy.%s" % (_t.__str__(), _t.__str__()))

# Type aliases
_type_aliases = [
    ('bool',    'bool'),
    ('int',     'int'),
    ('uint',    'numpy.uint64'),
    ('float',   'float'),
    ('complex', 'complex'),
    ('fmod',    'mod'),
    ('mod',     'remainder')
]

for _f, _t in _type_aliases:
    exec("%s = %s" % (_f, _t))

# Note that the following modules needs ufuncs and dtypes
from . import random123 as random

# Some modules (e.g. scipy) accesses '__all__' directly
__all__ = [x for x in dir() if not x.startswith("_")]

# Python v2 need '{set,get}_printoptions()' here
if sys.version_info[0] < 3:
    def set_printoptions(*args, **kwargs):
        numpy.core.arrayprint.set_printoptions(*args, **kwargs)
    def get_printoptions(*args, **kwargs):
        return numpy.core.arrayprint.get_printoptions()

# Let's bohriumify the exposed API
from . import bohriumify
bohriumify.modules()
