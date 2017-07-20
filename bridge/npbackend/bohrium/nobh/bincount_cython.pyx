import cython
from .. import array_create
import numpy_force as np
cimport numpy as cnp


@cython.boundscheck(False) # turn off bounds-checking
@cython.cdivision(True) # turn off division-by-zero checking
cdef _count(cnp.ndarray[cnp.uint64_t, ndim=1, mode='c'] x, cnp.ndarray[cnp.uint64_t, ndim=1, mode='c'] out):
    cdef int i
    for i in range(x.shape[0]):
        out[x[i]] += 1


def bincount_cython(x, minlength=None):
    """Cython/OpenMP implementation of `bincount()`"""

    x_max = int(x.max())
    if x_max < 0:
        raise RuntimeError("bincount(): first argument must be a 1 dimension, non-negative int array")
    if minlength is not None:
        x_max = max(x_max, minlength)

    x = array_create.array(x, bohrium=False, dtype=np.uint64)
    histogram = array_create.ones((x_max+1, ), dtype=x.dtype, bohrium=False)
    _count(x, histogram)
    histogram = array_create.array(histogram, bohrium=True)
    return histogram
