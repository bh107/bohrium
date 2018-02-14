"""
Signal Processing
~~~~~~~~~~~~~~~~~

Common signal processing functions, which often handle multiple dimension

"""
import numpy_force as numpy
from . import array_create
from . import bhary
from . import ufuncs
from . import linalg
from . import summations
from . import numpy_backport
from . import _bh


# 1d
# ---------------------------------------------------------------------------------
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
    s = numpy_backport.as_strided(padded, shape=(padded.shape[0] - filter.size + 1, filter.size),
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
        return numpy.convolve(a, v, mode)
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


# Nd
# ---------------------------------------------------------------------------------
def _findIndices(ArrSize, FilterSize):
    N = FilterSize.shape[0]
    n = int(FilterSize.prod())
    CumSizeArr = numpy.ones([N], dtype=numpy.int32)
    CumSizeArr[1:N] = ArrSize[0:N - 1].cumprod()
    CumSize = numpy.ones([N], dtype=numpy.int32)
    CumSize[1:N] = FilterSize[0:N - 1].cumprod()

    vals = numpy.empty((n, N), dtype=numpy.int32)
    for i in range(N):
        vals[:, i] = numpy.linspace(0, n - 1, n)

    vals = vals // CumSize
    vals = vals % FilterSize
    CurrPos = summations.sum(vals * CumSizeArr, axis=1)

    return CurrPos.astype(numpy.int32)


def _addZerosNd(Array, FilterSize, dtype):
    # Introduces zero padding for Column major flattening
    PaddedSize = numpy.array(Array.shape, dtype=numpy.int32)
    N = FilterSize.shape[0]
    PaddedSize[0:N] += FilterSize - 1
    cut = '['
    for i in range(PaddedSize.shape[0]):
        if i < N:
            minpos = int(FilterSize[i] / 2)
            maxpos = Array.shape[i] + int(FilterSize[i] / 2)
        else:
            minpos = 0
            maxpos = Array.shape[i]
        cut += str(minpos) + ':' + str(maxpos) + ','
    cut = cut[:-1] + ']'
    Padded = array_create.zeros(PaddedSize, dtype=dtype, bohrium=bhary.check(Array))
    exec ('Padded' + cut + '=Array')
    return Padded


def _findSame(Array, FilterSize):
    # Numpy convention. Returns view of the same size as the largest input
    N = FilterSize.shape[0]
    cut = '['
    for i in range(len(Array.shape)):
        if i < N:
            minpos = (FilterSize[i] - 1) // 2
            maxpos = Array.shape[i] - (FilterSize[i]) // 2
        else:
            minpos = 0
            maxpos = Array.shape[i]
        cut += str(minpos) + ':' + str(maxpos) + ','
    cut = cut[:-1] + ']'
    res = eval('Array' + cut)
    return res


def _findValid(Array, FilterSize):
    # Cuts the result down to only totally overlapping views
    N = FilterSize.shape[0]
    cut = '['
    for i in range(len(Array.shape)):
        if i < N:
            minpos = FilterSize[i] - 1
            maxpos = Array.shape[i] - FilterSize[i] + 1
        else:
            minpos = 0
            maxpos = Array.shape[i]
        cut += str(minpos) + ':' + str(maxpos) + ','
    cut = cut[:-1] + ']'
    res = eval('Array' + cut)
    return res


def _invert_ary(a):
    """Reverse all elements in each axis"""

    def flip(m, axis):
        """Copy of `numpy.flip()`, which were introduced in NumPy v1.12"""
        indexer = [slice(None)] * m.ndim
        try:
            indexer[axis] = slice(None, None, -1)
        except IndexError:
            raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                             % (axis, m.ndim))
        return m[tuple(indexer)]

    # Flip all axises
    for i in range(len(a.shape)):
        a = flip(a, axis=i)
    return a


def _correlate_kernel(Array, Filter, mode):
    # Anything to do?
    if Array.size <= 0:
        return Array.copy()

    # Complex correlate includes a conjugation
    if numpy.iscomplexobj(Filter):
        Filter = numpy.conj(Filter)

    # Get sizes as arrays for easier manipulation
    ArrSize = numpy.array(Array.shape, dtype=numpy.int32)
    FilterSize = numpy.array(Filter.shape, dtype=numpy.int32)

    # Check that mode='valid' is allowed given the array sizes
    if mode == 'valid':
        diffSize = ArrSize[:FilterSize.size] - FilterSize
        nSmaller = summations.sum(diffSize < 0)
        if nSmaller > 0:
            raise ValueError(
                "correlateNd: For 'valid' mode, one must be at least as large as the other in every dimension")

    # Use numpy convention for result dype
    dtype = numpy.result_type(Array, Filter)

    # Add zeros along relevant dimensions
    Padded = _addZerosNd(Array, FilterSize, dtype)
    PaddedSize = numpy.array(Padded.shape, dtype=numpy.int32)

    # Get positions of first view
    IndiVec = _findIndices(PaddedSize, FilterSize)
    CenterPos = tuple((FilterSize - 1) // 2)
    IndiMat = IndiVec.reshape(FilterSize, order='F')
    nPre = IndiMat[CenterPos]  # Required zeros before Array for correct alignment
    nPost = IndiVec[Filter.size - 1] - nPre
    n = Padded.size
    nTot = n + nPre + nPost  # Total size after pre/post padding
    V = array_create.empty([nTot], dtype=dtype, bohrium=bhary.check(Array))
    V[nPre:n + nPre] = Padded.flatten(order='F')
    V[:nPre] = 0
    V[n + nPre:] = 0
    A = Filter.flatten(order='F')

    # Actual correlation calculation
    Correlated = V[IndiVec[0]:n + IndiVec[0]] * A[0]
    for i in range(1, Filter.size):
        Correlated += V[IndiVec[i]:n + IndiVec[i]] * A[i]
        # TODO: we need this flush because of very slow fusion
        if bhary.check(V):
            _bh.flush()

    Full = Correlated.reshape(PaddedSize, order='F')
    if mode == 'full':
        return Full
    elif mode == 'same':
        return _findSame(Full, FilterSize)
    elif mode == 'valid':
        return _findValid(Full, FilterSize)
    else:
        raise ValueError("correlateNd: invalid mode '%s'" % mode)


def convolve(a, v, mode='full'):
    if (a.size > v.size) or (mode == 'same'):
        Array = a
        Filter = _invert_ary(v)
    else:
        Array = v
        Filter = _invert_ary(a)
    return _correlate_kernel(Array, Filter, mode)


def correlate(a, v, mode='valid'):
    if (a.size > v.size) or (mode == 'same'):
        Array = a
        Filter = v
    else:
        Array = _invert_ary(v)
        Filter = _invert_ary(a)
    return _correlate_kernel(Array, Filter, mode)
