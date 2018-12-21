"""
The module initialization of npbackend/bohrium imports and exposes all methods
required to become a drop-in replacement for numpy.
"""
import sys

if 'numpy_force' not in sys.modules:
    import numpy

    sys.modules['numpy_force'] = numpy
    del numpy

# We import all of NumPy and overwrite with the objects we implement our self
from numpy_force import *
from numpy_force import testing
import warnings
import bohrium_api
from .version import __version__, __version_info__
from .array_create import *
from .array_manipulation import *
from .reorganization import *
from .contexts import *
from .ufuncs import UFUNCS
from .masking import *
from .bhary import check, fix_biclass
from bohrium_api._info import numpy_types
from ._util import is_scalar
from . import linalg
from .linalg import matmul, dot, tensordot
from .summations import *
from .disk_io import *
from .concatenate import *
from .ufuncs import _handle__array_ufunc__
from . import contexts
from . import interop_pyopencl
from . import interop_pycuda
from . import interop_numpy
from . import backend_messaging
from . import loop
from . import user_kernel
from .loop import do_while
from .nobh import bincount
from .contexts import EnableBohrium as Enable, DisableBohrium as Disable
from ._bh import flush

# In NumPy `correlate` and `convolve` only handles 1D arrays whereas in SciPy they handles ND arrays.
# However, NumPy and SciPy's functionality differ! Thus, the ND version cannot replace NumPy's 1D version.
from .signal import correlate1d as correlate, convolve1d as convolve
from .signal import correlate as correlate_scipy, convolve as convolve_scipy
from numpy_force import dtype

asarray = array
asanyarray = array

if __version_info__ > bohrium_api.__version_info__:
    warnings.warn("The version of Bohrium API (%s) is newer than Bohrium (%s). Please upgrade "
                  "Bohrium (e.g. `pip install bohrium --upgrade`)" % (__version__, bohrium_api.__version__))


def replace_numpy(function):
    def wrapper(*args, **kwargs):
        with contexts.EnableBohrium():
            # Run your function/program
            result = function(*args, **kwargs)
        return result

    return wrapper


# Expose all ufuncs
for _name, _f in UFUNCS.items():
    if _name != "identity":  # NumPy has a function named `identity`, which we shouldn't overwrite
        exec ("%s = _f" % _name)

# Aliases
_aliases = [
    ('abs', 'absolute'),
    ('round', 'round_'),
    ('conjugate', 'conj'),
    # We handle all of NumPy's max and min the same!
    # Since reductions in OpenMP ignores NaN values we do the same always
    ('fmin', 'minimum'),
    ('fmax', 'maximum'),
    ('nanmin', 'minimum.reduce'),
    ('nanmax', 'maximum.reduce'),
    ('amin', 'minimum.reduce'),
    ('amax', 'maximum.reduce'),
]
for _f, _t in _aliases:
    exec ("%s = %s" % (_f, _t))
cumsum = add.accumulate
cumprod = multiply.accumulate

# Expose all data types
for _t in numpy_types():
    exec ("%s = numpy.%s" % (_t.__str__(), _t.__str__()))

# Type aliases
_type_aliases = [
    ('bool', 'bool'),
    ('int', 'int'),
    ('uint', 'numpy.uint64'),
    ('float', 'float'),
    ('complex', 'complex'),
    ('fmod', 'mod'),
    ('mod', 'remainder')
]

for _f, _t in _type_aliases:
    exec ("%s = %s" % (_f, _t))

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
