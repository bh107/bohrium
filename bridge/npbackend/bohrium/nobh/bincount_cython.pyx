from .. import array_create
from .._util import flush
from ..interop_numpy import get_array

import cython
from cython.parallel import prange, parallel
import numpy_force as np
cimport numpy as cnp

from libc.stdlib cimport abort, malloc, free
from libc.stdio cimport printf

ctypedef cnp.uint64_t uint64

IF UNAME_SYSNAME != "Darwin":
    cimport openmp


@cython.boundscheck(False) # turn off bounds-checking
@cython.cdivision(True) # turn off division-by-zero checking
cdef _count(uint64[:] x, uint64[:] out):
    cdef int num_threads, thds_id
    cdef uint64 i, start, end
    cdef uint64* local_histo

    IF UNAME_SYSNAME != "Darwin": # TODO: parallelize on OSX
        with nogil, parallel():
            num_threads = openmp.omp_get_num_threads()
            thds_id = openmp.omp_get_thread_num()
            start = (x.shape[0] / num_threads) * thds_id
            if thds_id == num_threads-1:
                end = x.shape[0]
            else:
                end = start + (x.shape[0] / num_threads)

            if not(thds_id < num_threads-1 and x.shape[0] < num_threads):
                local_histo = <uint64 *> malloc(sizeof(uint64) * out.shape[0])
                if local_histo == NULL:
                    abort()
                for i in range(out.shape[0]):
                    local_histo[i] = 0

                for i in range(start, end):
                    local_histo[x[i]] += 1

                with gil:
                    for i in range(out.shape[0]):
                        out[i] += local_histo[i]
                free(local_histo)
    ELSE:
        for i in range(out.shape[0]):
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
