import cython
from .. import array_create
from .._util import flush
import numpy_force as np
cimport numpy as cnp
from ..interop_numpy import get_array


@cython.boundscheck(False) # turn off bounds-checking
@cython.cdivision(True) # turn off division-by-zero checking
cdef _count(cnp.uint64_t[:] x, cnp.uint64_t[:] out):
    cdef int i
    for i in range(x.shape[0]):
        out[x[i]] += 1


def bincount_cython(x, minlength=None):
    """Cython/OpenMP implementation of `bincount()`"""

    x_max = int(x.max())
    if x_max < 0:
        raise RuntimeError("bincount(): first argument must be a 1 dimensional, non-negative int array")
    if minlength is not None:
        x_max = max(x_max, minlength)

    flush()
    x = array_create.array(x, dtype=np.uint64)
    ret = array_create.zeros((x_max+1, ), dtype=x.dtype)
    _count(get_array(x), get_array(ret))
    return ret
