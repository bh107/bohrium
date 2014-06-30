cimport numpy as np
from libc.stdint cimport uint64_t

cdef extern from "random123.h":
    uint64_t r123_ph2x32(uint64_t c, uint64_t k)

def ph2x32(np.uint64_t start_index, np.uint64_t key, object size):
    cdef np.uint64_t *array_data
    cdef np.ndarray array "arrayObject"
    cdef long length
    cdef long i

    if size is None:
        return r123_ph2x32(start_index,key)
    else:
        array = <np.ndarray>np.empty(size, np.uint64)
        length = np.PyArray_SIZE(array)
        array_data = <np.uint64_t *>array.data
        for i from 0 <= i < length:
            array_data[i] = r123_ph2x32(start_index,key)
            start_index += 1
        return array

