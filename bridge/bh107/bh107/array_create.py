"""
Array Creation Routines
=======================
"""
import math
import numpy as np
from . import bharray, _dtype_util
from bohrium_api import _bh_api, _info

__all__ = ['array', 'empty', 'zeros', 'ones', 'empty_like', 'zeros_like', 'ones_like', 'arange']


def array(obj, dtype=None, copy=False):
    """
    Create an BhArray.

    Parameters
    ----------
    obj : array_like
        An array, any object exposing the array interface, an
        object whose __array__ method returns an array, or any
        (nested) sequence.
    dtype : data-type, optional
        The desired data-type for the array.  If not given, then
        the type will be determined as the minimum type required
        to hold the objects in the sequence.  This argument can only
        be used to 'upcast' the array.  For downcasting, use the
        .astype(t) method.
    copy : bool, optional
        If true, then the object is copied.  Otherwise, a copy
        will only be made if obj isn't a BhArray of the correct dtype already

    Returns
    -------
    out : BhArray
        An array of dtype.

    See Also
    --------
    empty, empty_like, zeros, zeros_like, ones, ones_like, fill

    Examples
    --------
    >>> bh.array([1, 2, 3])
    array([1, 2, 3])

    Upcasting:

    >>> bh.array([1, 2, 3.0])
    array([ 1.,  2.,  3.])

    More than one dimension:

    >>> bh.array([[1, 2], [3, 4]])
    array([[1, 2],
           [3, 4]])

    Type provided:

    >>> bh.array([1, 2, 3], dtype=complex)
    array([ 1.+0.j,  2.+0.j,  3.+0.j])
    """

    if isinstance(obj, bharray.BhArray):
        if dtype is None:
            dtype = obj.dtype
        return obj.astype(dtype, always_copy=copy)
    else:
        return bharray.BhArray.from_object(obj)


def empty(shape, dtype=np.float64):
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
    """

    return bharray.BhArray(shape, dtype=dtype)


def zeros(shape, dtype=float):
    """
    Array of zeros.

    Return an array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : {sequence of ints, int}
        Shape of the array
    dtype : data-type, optional
        The desired data-type for the array, default is np.float64.

    Returns
    -------
    out : bharray
        Array of zeros of given shape, dtype, and order.
    """
    ret = empty(shape, dtype)
    ret.fill(0)
    return ret


def ones(shape, dtype=np.float64):
    """
    Array of ones.

    Return an array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : {sequence of ints, int}
        Shape of the array
    dtype : data-type, optional
        The desired data-type for the array, default is np.float64.

    Returns
    -------
    out : bharray
        Array of ones of given shape, dtype, and order.
    """
    ret = empty(shape, dtype)
    ret.fill(1)
    return ret


def empty_like(a, dtype=None):
    """
    Return a new array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of the
        returned array.
    dtype : data-type, optional
        Overrides the data type of the result.

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
    >>> bh.empty_like(a)
    array([[-1073741821, -1073741821,           3],    #random
           [          0,           0, -1073741821]])
    >>> a = np.array([[1., 2., 3.],[4.,5.,6.]])
    >>> bh.empty_like(a)
    array([[ -2.00000715e+000,   1.48219694e-323,  -2.00000572e+000],#random
           [  4.38791518e-305,  -2.00000715e+000,   4.17269252e-309]])
    """
    if dtype is None:
        dtype = a.dtype
    return empty(a.shape, dtype)


def zeros_like(a, dtype=None):
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
    >>> x = bh.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> bh.zeros_like(x)
    array([[0, 0, 0],
           [0, 0, 0]])

    >>> y = bh.arange(3, dtype=bh.float)
    >>> y
    array([ 0.,  1.,  2.])
    >>> bh.zeros_like(y)
    array([ 0.,  0.,  0.])

    """
    ret = empty_like(a, dtype=dtype)
    ret.fill(0)
    return ret


def ones_like(a, dtype=None):
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
    >>> x = bh.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> bh.ones_like(x)
    array([[1, 1, 1],
           [1, 1, 1]])

    >>> y = bh.arange(3, dtype=bh.float)
    >>> y
    array([ 0.,  1.,  2.])
    >>> bh.ones_like(y)
    array([ 1.,  1.,  1.])

    """
    ret = empty_like(a, dtype=dtype)
    ret.fill(0)
    return ret


def simply_range(size, dtype=np.uint64):
    if not isinstance(size, _dtype_util.integers):
        raise ValueError("size must be an integer")

    if size < 1:
        raise ValueError("size must be greater than 0")

    if _dtype_util.size_of(dtype) > 4:
        ret = empty((size,), dtype=np.uint64)
    else:
        ret = empty((size,), dtype=np.uint32)

    _bh_api.op(_info.op["range"]["id"], [_dtype_util.np2bh_enum(ret.dtype)], [ret._bhc_handle])
    return ret.astype(dtype, always_copy=False)


def arange(start, stop=None, step=1, dtype=None):
    """
    arange([start,] stop[, step,], dtype=None)

    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range <http://docs.python.org/lib/built-in-funcs.html>`_ function,
    but returns a ndarray rather than a list.

    When using a non-integer step, such as 0.1, the results will often not
    be consistent.  It is better to use ``linspace`` for these cases.

    Parameters
    ----------
    start : number, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : number
        End of interval.  The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : number, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified, `start` must also be given.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.

    Returns
    -------
    out : ndarray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.

    See Also
    --------
    linspace : Evenly spaced numbers with careful handling of endpoints.
    ogrid: Arrays of evenly spaced numbers in N-dimensions
    mgrid: Grid-shaped arrays of evenly spaced numbers in N-dimensions

    Examples
    --------
    >>> bh.arange(3)
    array([0, 1, 2])
    >>> bh.arange(3.0)
    array([ 0.,  1.,  2.])
    >>> bh.arange(3,7)
    array([3, 4, 5, 6])
    >>> bh.arange(3,7,2)
    array([3, 5])

    """
    if stop is None:
        stop = start
        start = type(stop)(0)

    if not (isinstance(stop, _dtype_util.integers) and isinstance(start, _dtype_util.integers)):
        raise ValueError("arange(): start and stop must be integers")

    if step == 0:
        raise ValueError("arange(): step cannot be zero")

    # Let's make sure that 'step' is always positive
    swap_back = False
    if step < 0:
        step *= -1
        (start, stop) = (stop, start)
        swap_back = True

    if start >= stop:
        return empty(tuple(), dtype=dtype)

    size = int(math.ceil((float(stop) - float(start)) / float(step)))
    if dtype is None:
        dtype = np.int64

    result = simply_range(size, dtype=dtype)
    if swap_back:
        step *= -1
        (start, stop) = (stop, start)

    if step != 1:
        result *= step

    if start != 0:
        result += start
    return result
