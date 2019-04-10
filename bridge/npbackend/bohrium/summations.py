"""
Summations and products
~~~~~~~~~~~~~~~~~~~~~~~
"""
import warnings
from . import ufuncs
from . import array_create
from . import bhary
from . import reorganization
from . import array_manipulation
import numpy_force as numpy


@bhary.fix_biclass_wrapper
def sum(a, axis=None, dtype=None, out=None):
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
    dtype : dtype, optional
        The type of the returned array and of the accumulator in which the
        elements are summed.  The dtype of `a` is used by default unless `a`
        has an integer dtype of less precision than the default platform
        integer.  In that case, if `a` is signed then the platform integer
        is used while if `a` is unsigned then an unsigned integer of the
        same precision as the platform integer is used.
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

    if not bhary.check(a) and not bhary.check(out):
        return numpy.sum(a, axis=axis, dtype=dtype, out=out)
    else:
        if dtype is not None:
            a = array_create.array(a, dtype=dtype)
        return ufuncs.add.reduce(a, axis=axis, out=out)


@bhary.fix_biclass_wrapper
def prod(a, axis=None, dtype=None, out=None):
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
    dtype : dtype, optional
        The type of the returned array and of the accumulator in which the
        elements are summed.  The dtype of `a` is used by default unless `a`
        has an integer dtype of less precision than the default platform
        integer.  In that case, if `a` is signed then the platform integer
        is used while if `a` is unsigned then an unsigned integer of the
        same precision as the platform integer is used.
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

    if not bhary.check(a) and not bhary.check(out):
        return numpy.prod(a, axis=axis, dtype=dtype, out=out)
    else:
        if dtype is not None:
            a = array_create.array(a, dtype=dtype)
        return ufuncs.multiply.reduce(a, axis=axis, out=out)


@bhary.fix_biclass_wrapper
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

    if not bhary.check(a) and not bhary.check(out):
        return numpy.max(a, axis=axis, out=out)  # NumPy 1.6 doesn't support axis=None
    else:
        return ufuncs.maximum.reduce(a, axis=axis, out=out)


@bhary.fix_biclass_wrapper
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

    if not bhary.check(a) and not bhary.check(out):
        return numpy.min(a, axis=axis, out=out)  # NumPy 1.6 doesn't support axis=None
    else:
        return ufuncs.minimum.reduce(a, axis=axis, out=out)


@bhary.fix_biclass_wrapper
def any(a, axis=None, out=None, keepdims=None):
    if not bhary.check(a) and not bhary.check(out):
        return numpy.any(a, axis=axis, out=out)
    else:
        return ufuncs.logical_or.reduce(a.astype(bool), axis=axis, out=out)


@bhary.fix_biclass_wrapper
def all(a, axis=None, out=None, keepdims=None):
    if not bhary.check(a) and not bhary.check(out):
        return numpy.all(a, axis=axis, out=out)
    else:
        return ufuncs.logical_and.reduce(a.astype(bool), axis=axis, out=out)


@bhary.fix_biclass_wrapper
def argmax(a, axis=None, out=None):
    """
    Returns the indices of the maximum values along an axis.
    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.
    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.
    See Also
    --------
    ndarray.argmax, argmin
    amax : The maximum value along a given axis.
    unravel_index : Convert a flat index into an index tuple.
    Notes
    -----
    In case of multiple occurrences of the maximum values, the indices
    corresponding to the first occurrence are returned.
    Examples
    --------
    >>> a = np.arange(6).reshape(2,3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.argmax(a)
    5
    >>> np.argmax(a, axis=0)
    array([1, 1, 1])
    >>> np.argmax(a, axis=1)
    array([2, 2])
    >>> b = np.arange(6)
    >>> b[1] = 5
    >>> b
    array([0, 5, 2, 3, 4, 5])
    >>> np.argmax(b) # Only the first occurrence is returned.
    1
    """

    if not bhary.check(a):
        return numpy.argmax(a, axis=axis, out=out)

    if axis is None or (a.ndim == 1 and axis == 0):
        a = array_manipulation.flatten(a, always_copy=False)
        ret = reorganization.flatnonzero(a == max(a))[0]
    else:
        warnings.warn("Bohrium does not support the 'axis' argument, "
                      "it will be handled by the original NumPy.", UserWarning, 2)
        return numpy.argmax(a.copy2numpy(), axis=axis)

    if out is None:
        return ret
    else:
        out[...] = ret
        return out


@bhary.fix_biclass_wrapper
def argmin(a, axis=None, out=None):
    """
    Returns the indices of the minimum values along an axis.
    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.
    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.
    See Also
    --------
    ndarray.argmin, argmax
    amin : The minimum value along a given axis.
    unravel_index : Convert a flat index into an index tuple.
    Notes
    -----
    In case of multiple occurrences of the minimum values, the indices
    corresponding to the first occurrence are returned.
    Examples
    --------
    >>> a = np.arange(6).reshape(2,3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.argmin(a)
    0
    >>> np.argmin(a, axis=0)
    array([0, 0, 0])
    >>> np.argmin(a, axis=1)
    array([0, 0])
    >>> b = np.arange(6)
    >>> b[4] = 0
    >>> b
    array([0, 1, 2, 3, 0, 5])
    >>> np.argmin(b) # Only the first occurrence is returned.
    0
    """

    if not bhary.check(a):
        return numpy.argmin(a, axis=axis, out=out)

    if axis is None or (a.ndim == 1 and axis == 0):
        a = array_manipulation.flatten(a, always_copy=False)
        ret = reorganization.flatnonzero(a == min(a))[0]
    else:
        warnings.warn("Bohrium does not support the 'axis' argument, "
                      "it will be handled by the original NumPy.", UserWarning, 2)
        return numpy.argmin(a.copy2numpy(), axis=axis)

    if out is None:
        return ret
    else:
        out[...] = ret
        return out


def mean(a, axis=None, dtype=None, out=None):
    """
    Compute the arithmetic mean along the specified axis.
    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.
    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.
        .. versionadded:: 1.7.0
        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for floating point inputs, it is the same as the
        input dtype.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
        See `doc.ufuncs` for details.

    Returns
    -------
    m : ndarray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned.
    See Also
    --------
    average : Weighted average
    std, var, nanmean, nanstd, nanvar
    Notes
    -----
    The arithmetic mean is the sum of the elements along the axis divided
    by the number of elements.
    Note that for floating-point input, the mean is computed using the
    same precision the input has.  Depending on the input data, this can
    cause the results to be inaccurate, especially for `float32` (see
    example below).  Specifying a higher-precision accumulator using the
    `dtype` keyword can alleviate this issue.
    By default, `float16` results are computed using `float32` intermediates
    for extra precision.
    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.mean(a)
    2.5
    >>> np.mean(a, axis=0)
    array([ 2.,  3.])
    >>> np.mean(a, axis=1)
    array([ 1.5,  3.5])
    In single precision, `mean` can be inaccurate:
    >>> a = np.zeros((2, 512*512), dtype=np.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> np.mean(a)
    0.54999924
    Computing the mean in float64 is more accurate:
    >>> np.mean(a, dtype=np.float64)
    0.55000000074505806
    """

    def _count_reduce_items(arr, axis):
        if axis is None:
            axis = tuple(range(arr.ndim))
        if not isinstance(axis, tuple):
            axis = (axis,)
        items = 1
        for ax in axis:
            items *= arr.shape[ax]
        return items

    def _mean(a, axis=None, dtype=None, out=None):
        arr = array_create.array(a)
        is_float16_result = False
        rcount = _count_reduce_items(arr, axis)
        # Make this warning show up first
        if rcount == 0:
            warnings.warn("Mean of empty slice.", RuntimeWarning, stacklevel=2)

        # Cast bool, unsigned int, and int to float64 by default
        if dtype is None:
            if issubclass(arr.dtype.type, (numpy.integer, numpy.bool_)):
                dtype = numpy.dtype('f8')
            elif issubclass(arr.dtype.type, numpy.float16):
                dtype = numpy.dtype('f4')
                is_float16_result = True

        ret = sum(arr, axis, dtype, out)
        if isinstance(ret, numpy.ndarray):
            ret = ufuncs.true_divide(ret, rcount, out=ret)
            if is_float16_result and out is None:
                ret = a.dtype.type(ret)
        elif hasattr(ret, 'dtype'):
            if is_float16_result:
                ret = a.dtype.type(ret / rcount)
            else:
                ret = ret.dtype.type(ret / rcount)
        else:
            ret = ret / rcount
        return ret
    return _mean(a, axis=axis, dtype=dtype, out=out)


average = mean
