"""
Summations and products
~~~~~~

Common linear algebra functions

"""
from . import ufunc
import numpy_force as numpy
from . import ndarray

def sum(a, axis=None, out=None):
    """
    Sum of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Elements to sum.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.
        The default (`axis` = `None`) is perform a sum over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.

        .. versionadded:: 1.7.0

        If this is a tuple of ints, a sum is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : ndarray, optional
        Array into which the output is placed.  By default, a new array is
        created.  If `out` is given, it must be of the appropriate shape
        (the shape of `a` with `axis` removed, i.e.,
        ``numpy.delete(a.shape, axis)``).  Its type is preserved. See
        `doc.ufuncs` (Section "Output arguments") for more details.

    Returns
    -------
    sum_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar
        is returned.  If an output array is specified, a reference to
        `out` is returned.

    See Also
    --------
    ndarray.sum : Equivalent method.

    cumsum : Cumulative sum of array elements.

    trapz : Integration of array values using the composite trapezoidal rule.

    mean, average

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    Examples
    --------
    >>> np.sum([0.5, 1.5])
    2.0
    >>> np.sum([0.5, 0.7, 0.2, 1.5], dtype=np.int32)
    1
    >>> np.sum([[0, 1], [0, 5]])
    6
    >>> np.sum([[0, 1], [0, 5]], axis=0)
    array([0, 6])
    >>> np.sum([[0, 1], [0, 5]], axis=1)
    array([1, 5])

    If the accumulator is too small, overflow occurs:

    >>> np.ones(128, dtype=np.int8).sum(dtype=np.int8)
    -128

    """

    if not ndarray.check(a) and not ndarray.check(out):
        return numpy.sum(a, axis=axis, out=out)#NumPy 1.6 doesn't support axis=None
    else:
        return ufunc.add.reduce(a, axis=axis, out=out)

def prod(a, axis=None, out=None):
    """
    Product of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Elements to multiply.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a multiply is performed.
        The default (`axis` = `None`) is perform a multiply over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.

        .. versionadded:: 1.7.0

        If this is a tuple of ints, a multiply is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : ndarray, optional
        Array into which the output is placed.  By default, a new array is
        created.  If `out` is given, it must be of the appropriate shape
        (the shape of `a` with `axis` removed, i.e.,
        ``numpy.delete(a.shape, axis)``).  Its type is preserved. See
        `doc.ufuncs` (Section "Output arguments") for more details.

    Returns
    -------
    protuct_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar
        is returned.  If an output array is specified, a reference to
        `out` is returned.

    Examples
    --------
    >>> np.prod([0.5, 1.5])
    2.0
    >>> np.prod([0.5, 0.7, 0.2, 1.5], dtype=np.int32)
    1
    >>> np.prod([[0, 1], [0, 5]])
    6
    >>> np.prod([[0, 1], [0, 5]], axis=0)
    array([0, 6])
    >>> np.prod([[0, 1], [0, 5]], axis=1)
    array([1, 5])

    If the accumulator is too small, overflow occurs:

    >>> np.ones(128, dtype=np.int8).prod(dtype=np.int8)
    -128

    """

    if not ndarray.check(a) and not ndarray.check(out):
        return numpy.prod(a, axis=axis, out=out)#NumPy 1.6 doesn't support axis=None
    else:
        return ufunc.multiply.reduce(a, axis=axis, out=out)

def max(a, axis=None, out=None):
    """
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a max is performed.
        The default (`axis` = `None`) is perform a max over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.
        If this is a tuple of ints, a max is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See `doc.ufuncs` (Section "Output arguments") for more details.

    Returns
    -------
    max : ndarray or scalar
        Maximum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is given, the result is an array of dimension
        ``a.ndim - 1``.

    See Also
    --------
    min :
        The minimum value of an array along a given axis, propagating any NaNs.
    nanmax :
        The maximum value of an array along a given axis, ignoring any NaNs.
    maximum :
        Element-wise maximum of two arrays, propagating any NaNs.
    fmax :
        Element-wise maximum of two arrays, ignoring any NaNs.
    argmax :
        Return the indices of the maximum values.

    nanmin, minimum, fmin

    Notes
    -----
    NaN values are propagated, that is if at least one item is NaN, the
    corresponding max value will be NaN as well. To ignore NaN values
    (MATLAB behavior), please use nanmax.

    Don't use `max` for element-wise comparison of 2 arrays; when
    ``a.shape[0]`` is 2, ``maximum(a[0], a[1])`` is faster than
    ``max(a, axis=0)``.

    Examples
    --------
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> np.max(a)           # Maximum of the flattened array
    3
    >>> np.max(a, axis=0)   # Maxima along the first axis
    array([2, 3])
    >>> np.max(a, axis=1)   # Maxima along the second axis
    array([1, 3])

    >>> b = np.arange(5, dtype=np.float)
    >>> b[2] = np.NaN
    >>> np.max(b)
    nan
    >>> np.nanmax(b)
    4.0

    """

    if not ndarray.check(a) and not ndarray.check(out):
        return numpy.max(a, axis=axis, out=out)#NumPy 1.6 doesn't support axis=None
    else:
        return ufunc.maximum.reduce(a, axis=axis, out=out)


def min(a, axis=None, out=None):
    """
    Return the minimum of an array or minimum along an axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a min is performed.
        The default (`axis` = `None`) is perform a min over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.
        If this is a tuple of ints, a min is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See `doc.ufuncs` (Section "Output arguments") for more details.

    Returns
    -------
    min : ndarray or scalar
        minimum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is given, the result is an array of dimension
        ``a.ndim - 1``.

    See Also
    --------
    min :
        The minimum value of an array along a given axis, propagating any NaNs.
    nanmin :
        The minimum value of an array along a given axis, ignoring any NaNs.
    minimum :
        Element-wise minimum of two arrays, propagating any NaNs.
    fmin :
        Element-wise minimum of two arrays, ignoring any NaNs.
    argmin :
        Return the indices of the minimum values.

    nanmin, minimum, fmin

    Notes
    -----
    NaN values are propagated, that is if at least one item is NaN, the
    corresponding min value will be NaN as well. To ignore NaN values
    (MATLAB behavior), please use nanmin.

    Don't use `min` for element-wise comparison of 2 arrays; when
    ``a.shape[0]`` is 2, ``minimum(a[0], a[1])`` is faster than
    ``min(a, axis=0)``.

    Examples
    --------
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> np.min(a)           # minimum of the flattened array
    3
    >>> np.min(a, axis=0)   # minima along the first axis
    array([2, 3])
    >>> np.min(a, axis=1)   # minima along the second axis
    array([1, 3])

    >>> b = np.arange(5, dtype=np.float)
    >>> b[2] = np.NaN
    >>> np.min(b)
    nan
    >>> np.nanmin(b)
    4.0

    """

    if not ndarray.check(a) and not ndarray.check(out):
        return numpy.min(a, axis=axis, out=out)#NumPy 1.6 doesn't support axis=None
    else:
        return ufunc.minimum.reduce(a, axis=axis, out=out)
