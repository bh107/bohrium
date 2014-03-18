"""
Array Create Routines
~~~~

"""
import _bh
import _info
import _util
import ndarray
import numpy


def empty(shape, dtype=float):
    """
    Return a new matrix of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty matrix.
    dtype : data-type, optional
        Desired output data-type.

    See Also
    --------
    empty_like, zeros

    Notes
    -----
    The order of the data in memory is always row-major (C-style).

    `empty`, unlike `zeros`, does not set the matrix values to zero,
    and may therefore be marginally faster.  On the other hand, it requires
    the user to manually set all the values in the array, and should be
    used with caution.

    Examples
    --------
    >>> import numpy.matlib
    >>> np.matlib.empty((2, 2))    # filled with random data
    matrix([[  6.76425276e-320,   9.79033856e-307],
            [  7.39337286e-309,   3.22135945e-309]])        #random
    >>> np.matlib.empty((2, 2), dtype=int)
    matrix([[ 6600475,        0],
            [ 6586976, 22740995]])                          #random

    """
    ret = _bh.ndarray(shape, dtype=dtype)
    ndarray.new_bhc_base(ret)#Trigger Bohrium creations
    return ret

def array(object, dtype=None, ndmin=0, bohrium=True):
    """
    Create an Bohrium array. Will copy and use C-contiguous order always

    Parameters
    ----------
    object : array_like
        An array, any object exposing the array interface, an object
        whose __array__ method returns an array, or any (nested) sequence.
    dtype : data-type, optional
        The desired data-type for the array. If not given, then the type
        will be determined as the minimum type required to hold the objects
        in the sequence. This argument can only be used to 'upcast' the array.
        For downcasting, use the .astype(t) method.s
    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting array should have.
        Ones will be pre-pended to the shape as needed to meet this requirement.

    Returns
    -------
    out : ndarray
        An array object satisfying the specified requirements.
    """
    a = numpy.array(object, dtype=dtype, ndmin=ndmin)
    ret = empty(a.shape, dtype=dtype)
    ret._data_fill(a)
    return a


###############################################################################
################################ UNIT TEST ####################################
###############################################################################

import unittest

class Tests(unittest.TestCase):

    def test_empty_dtypes(self):
        for t in _info.numpy_types:
            a = empty((4,4), dtype=t)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
    unittest.TextTestRunner(verbosity=2).run(suite)
