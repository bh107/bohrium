"""
Signal Processing
~~~~~~~~~~~~~~~~~

Common signal processing functions, which often handle multiple dimension

"""
import numpy_force as numpy
from numpy_force.lib.stride_tricks import as_strided
from . import array_create
from . import bhary
from . import ufuncs
from . import linalg


def _correlate_and_convolve_body(vector, filter, d, mode):
    """ The body of correlate() and convolve() are identical"""

    # Anything to do?
    if vector.size <= 0:
        return vector.copy()

    # Complex correlate includes a conjugation
    if numpy.iscomplexobj(filter):
        filter = numpy.conj(filter)

    dtype = numpy.result_type(vector, filter)
    rows = int(ufuncs.ceil((vector.size + 2 * filter.size - 2) / float(filter.size)))
    padded = array_create.empty([rows * filter.size], dtype=dtype)
    padded[0:filter.size - 1] = 0
    padded[filter.size - 1:vector.size + filter.size - 1] = vector
    padded[vector.size + filter.size - 1:] = 0
    s = as_strided(padded, shape=(padded.shape[0] - filter.size + 1, filter.size),
                   strides=(padded.strides[0], padded.strides[0]))
    result = linalg.dot(s, filter)
    if mode == 'same':
        return result[d:vector.size + d]
    elif mode == 'full':
        return result[0:vector.size + filter.size - 1]
    elif mode == 'valid':
        return result[filter.size - 1:vector.size]
    else:
        raise ValueError("correlate1d: invalid mode '%s'" % mode)


def correlate1d(a, v, mode='valid'):
    assert a.ndim == 1
    assert v.ndim == 1

    # Let's make sure that we are working on Bohrium arrays
    if not bhary.check(a) and not bhary.check(v):
        return numpy.correlate(a, v, mode)
    else:
        a = array_create.array(a)
        v = array_create.array(v)

    # We might have to swap `a` and `v` such that `vector` is always largest
    if v.shape[0] > a.shape[0]:
        vector = v[::-1]
        filter = a[::-1]
        d = int(filter.size / 2)
    else:
        vector = a
        filter = v
        d = int((filter.size - 1) / 2)
    return _correlate_and_convolve_body(vector, filter, d, mode)


def convolve1d(a, v, mode='full'):
    assert a.ndim == 1
    assert v.ndim == 1

    # Let's make sure that we are working on Bohrium arrays
    if not bhary.check(a) and not bhary.check(v):
        return numpy.correlate(a, v, mode)
    else:
        a = array_create.array(a)
        v = array_create.array(v)

    # We might have to swap `a` and `v` such that `vector` is always largest
    if v.shape[0] > a.shape[0]:
        vector = v
        filter = a[::-1]
    else:
        vector = a
        filter = v[::-1]
    d = int((filter.size - 1) / 2)
    return _correlate_and_convolve_body(vector, filter, d, mode)
