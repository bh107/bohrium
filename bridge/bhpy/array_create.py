"""
Array Create Routines
~~~~

"""
import ndarray
import numpy

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
    if not bohrium:
        return a.copy()
    ret = empty(a.shape, dtype=a.dtype)
    ret._data_fill(a)
    return ret

def empty(shape, dtype=float, bohrium=True):
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
    if not bohrium:
        return numpy.empty(shape, dtype=dtype)
    return ndarray.new(shape, dtype)

def ones(shape, dtype=float, bohrium=True):
    """
    Matrix of ones.

    Return a matrix of given shape and type, filled with ones.

    Parameters
    ----------
    shape : {sequence of ints, int}
        Shape of the matrix
    dtype : data-type, optional
        The desired data-type for the matrix, default is np.float64.
    bohrium : boolean, optional
        Determines whether it is a Bohrium-enabled array or a regular NumPy array

    Returns
    -------
    out : matrix
        Matrix of ones of given shape, dtype, and order.

    See Also
    --------
    ones : Array of ones.
    matlib.zeros : Zero matrix.

    Notes
    -----
    The order of the data in memory is always row-major (C-style).

    If `shape` has length one i.e. ``(N,)``, or is a scalar ``N``,
    `out` becomes a single row matrix of shape ``(1,N)``.

    Examples
    --------
    >>> np.matlib.ones((2,3))
    matrix([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])

    >>> np.matlib.ones(2)
    matrix([[ 1.,  1.]])

    """
    A = empty(shape, dtype=dtype, bohrium=bohrium)
    A[:] = A.dtype.type(1)
    return A

def zeros(shape, dtype=float, bohrium=True):
    """
    Return a matrix of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the matrix
    dtype : data-type, optional
        The desired data-type for the matrix, default is float.
    bohrium : boolean, optional
        Determines whether it is a Bohrium-enabled array or a regular NumPy array

    Returns
    -------
    out : matrix
        Zero matrix of given shape, dtype, and order.

    See Also
    --------
    numpy.zeros : Equivalent array function.
    matlib.ones : Return a matrix of ones.

    Notes
    -----
    The order of the data in memory is always row-major (C-style).

    If `shape` has length one i.e. ``(N,)``, or is a scalar ``N``,
    `out` becomes a single row matrix of shape ``(1,N)``.

    Examples
    --------
    >>> import numpy.matlib
    >>> np.matlib.zeros((2, 3))
    matrix([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])

    >>> np.matlib.zeros(2)
    matrix([[ 0.,  0.]])

    """
    A = empty(shape, dtype=dtype, bohrium=bohrium)
    A[:] = A.dtype.type(0)
    return A

def empty_like(a, dtype=None, bohrium=None):
    """
    Return a new array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of the
        returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    bohrium : boolean, optional
        Determines whether it is a Bohrium-enabled array or a regular NumPy array

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data with the same
        shape and type as `a`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.

    Notes
    -----
    The order of the data in memory is always row-major (C-style).

    This function does *not* initialize the returned array; to do that use
    `zeros_like` or `ones_like` instead.  It may be marginally faster than
    the functions that do set the array values.

    Examples
    --------
    >>> a = ([1,2,3], [4,5,6])                         # a is array-like
    >>> np.empty_like(a)
    array([[-1073741821, -1073741821,           3],    #random
           [          0,           0, -1073741821]])
    >>> a = np.array([[1., 2., 3.],[4.,5.,6.]])
    >>> np.empty_like(a)
    array([[ -2.00000715e+000,   1.48219694e-323,  -2.00000572e+000],#random
           [  4.38791518e-305,  -2.00000715e+000,   4.17269252e-309]])
    """
    if dtype == None:
        dtype = a.dtype
    if bohrium == None:
        bohrium = ndarray.check(a)
    return empty(a.shape, dtype, bohrium)

def zeros_like(a, dtype=None, bohrium=None):
    """
    Return an array of zeros with the same shape and type as a given array.

    With default parameters, is equivalent to ``a.copy().fill(0)``.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    bohrium : boolean, optional
        Determines whether it is a Bohrium-enabled array or a regular NumPy array

    Returns
    -------
    out : ndarray
        Array of zeros with the same shape and type as `a`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.

    Notes
    -----
    The order of the data in memory is always row-major (C-style).

    Examples
    --------
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.zeros_like(x)
    array([[0, 0, 0],
           [0, 0, 0]])

    >>> y = np.arange(3, dtype=np.float)
    >>> y
    array([ 0.,  1.,  2.])
    >>> np.zeros_like(y)
    array([ 0.,  0.,  0.])

    """
    if dtype == None:
        dtype = a.dtype
    if bohrium == None:
        bohrium = ndarray.check(a)
    b = empty_like(a, dtype=dtype, bohrium=bohrium)
    b[:] = b.dtype.type(0)
    return b

def ones_like(a, dtype=None, bohrium=None):
    """
    Return an array of ones with the same shape and type as a given array.

    With default parameters, is equivalent to ``a.copy().fill(1)``.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    bohrium : boolean, optional
        Determines whether it is a Bohrium-enabled array or a regular NumPy array

    Returns
    -------
    out : ndarray
        Array of zeros with the same shape and type as `a`.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.

    Notes
    -----
    The order of the data in memory is always row-major (C-style).

    Examples
    --------
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.ones_like(x)
    array([[1, 1, 1],
           [1, 1, 1]])

    >>> y = np.arange(3, dtype=np.float)
    >>> y
    array([ 0.,  1.,  2.])
    >>> np.ones_like(y)
    array([ 1.,  1.,  1.])

    """
    if dtype == None:
        dtype = a.dtype
    if bohrium == None:
        bohrium = ndarray.check(a)
    b = empty_like(a, dtype=dtype, bohrium=bohrium)
    b[:] = b.dtype.type(1)
    return b



###############################################################################
################################ UNIT TEST ####################################
###############################################################################

import unittest
import _info

class Tests(unittest.TestCase):

    def test_empty_dtypes(self):
        for t in _info.numpy_types:
            a = empty((4,4), dtype=t)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
    unittest.TextTestRunner(verbosity=2).run(suite)
