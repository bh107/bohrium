from .. import array_create
from .._util import flush, dtype_equal
from ..interop_numpy import get_array
from ..bhary import get_base

import cython
from cython.parallel import prange, parallel
import numpy_force as np

from libc.stdlib cimport abort, malloc, free
cimport numpy as cnp
ctypedef cnp.uint64_t uint64
ctypedef cnp.int64_t int64

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
        with nogil:
            for i in range(x.shape[0]):
                out[x[i]] += 1


@cython.boundscheck(False) # turn off bounds-checking
@cython.cdivision(True) # turn off division-by-zero checking
cdef _count_int_weights(uint64[:] x, int64[:] w, int64[:] out):
    cdef int num_threads, thds_id
    cdef uint64 i, start, end
    cdef int64* local_histo

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
                local_histo = <int64 *> malloc(sizeof(int64) * out.shape[0])
                if local_histo == NULL:
                    abort()
                for i in range(out.shape[0]):
                    local_histo[i] = 0

                for i in range(start, end):
                    local_histo[x[i]] += w[i]

                with gil:
                    for i in range(out.shape[0]):
                        out[i] += local_histo[i]
                free(local_histo)
    ELSE:
        with nogil:
            for i in range(x.shape[0]):
                out[x[i]] += w[i]


@cython.boundscheck(False) # turn off bounds-checking
@cython.cdivision(True) # turn off division-by-zero checking
cdef _count_float_weights(uint64[:] x, double[:] w, double[:] out):
    cdef int num_threads, thds_id
    cdef uint64 i, start, end
    cdef double* local_histo

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
                local_histo = <double *> malloc(sizeof(double) * out.shape[0])
                if local_histo == NULL:
                    abort()
                for i in range(out.shape[0]):
                    local_histo[i] = 0

                for i in range(start, end):
                    local_histo[x[i]] += w[i]

                with gil:
                    for i in range(out.shape[0]):
                        out[i] += local_histo[i]
                free(local_histo)
    ELSE:
        with nogil:
            for i in range(x.shape[0]):
                out[x[i]] += w[i]


def bincount_cython(x, weights=None, minlength=None):
    """Cython/OpenMP implementation of `bincount()`"""

    x_max = int(x.max())
    if x_max < 0:
        raise RuntimeError("bincount(): first argument must be a 1 dimensional, non-negative int array")
    if minlength is not None:
        x_max = max(x_max, minlength)

    x = array_create.array(x, dtype=np.uint64)
    flush()
    if weights is None:
        ret = array_create.zeros((x_max+1, ), dtype=x.dtype)
        _count(get_array(x), get_array(ret))
    else:
        if np.issubdtype(weights.dtype, np.integer):
            weights = array_create.array(weights, dtype=np.int64)
            ret = array_create.zeros((x_max+1, ), dtype=weights.dtype)
            _count_int_weights(get_array(x), get_array(weights), get_array(ret))
        elif dtype_equal(weights.dtype, np.float32) or dtype_equal(weights.dtype, np.float64):
            weights = array_create.array(weights, dtype=np.float64)
            ret = array_create.zeros((x_max+1, ), dtype=weights.dtype)
            _count_float_weights(get_array(x), get_array(get_base(weights)), get_array(ret))
        else:
            raise RuntimeError("bincount(): weights has unsupported dtype (%s)" % weights.dtype)
    return ret
