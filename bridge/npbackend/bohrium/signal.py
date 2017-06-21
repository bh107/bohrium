"""
Signal Processing
~~~~~~~~~~~~~~~~~

Common signal processing functions, which often handle multiple dimension

"""
import bohrium as np
import numpy_force as numpy
from numpy_force.lib.stride_tricks import as_strided


def correlate1d(a, v, mode='valid'):
    assert a.ndim == 1
    assert v.ndim == 1

    # Let's make sure that we are working on Bohrium arrays
    if not np.check(a) and not np.check(v):
        return numpy.correlate(a, v, mode)
    else:
        a = np.array(a)
        v = np.array(v)

    # We might have to swap `a` and `v` such at `a` `vector` is always largest
    if v.shape[0] > a.shape[0]:
        filter_size = a.shape[0]
        vector_size = v.shape[0]
        vector = v[::-1]
        filter = a[::-1]
        d = int(filter_size / 2)
    else:
        filter_size = v.shape[0]
        vector_size = a.shape[0]
        vector = a
        filter = v
        d = int((filter_size - 1) / 2)

    # Complex correlate includes a conjugation
    if numpy.iscomplexobj(filter):
        filter = np.conj(filter)

    # Anything to do?
    if vector.size <= 0:
        return vector.copy()

    dtype = numpy.result_type(a, v)
    rows = int(np.ceil((vector_size + 2 * filter_size - 2) / float(filter_size)))
    padded = np.empty([rows * filter_size], dtype=dtype)
    padded[0:filter_size - 1] = 0
    padded[filter_size - 1:vector_size + filter_size - 1] = vector
    padded[vector_size + filter_size - 1:] = 0
    s = as_strided(padded, shape=(padded.shape[0] - filter_size + 1, filter_size),
                   strides=(padded.strides[0], padded.strides[0]))
    result = np.dot(s, filter)
    if mode == 'same':
        return result[d:vector_size + d]
    elif mode == 'full':
        return result[0:vector_size + filter_size - 1]
    elif mode == 'valid':
        return result[filter_size - 1:vector_size]
    else:
        raise ValueError("correlate1d: invalid mode '%s'" % mode)


def convolve1d(a, v, mode='full'):
    return correlate1d(a, v[::-1], mode=mode)
