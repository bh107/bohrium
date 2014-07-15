"""
Summations and products
~~~~~~

Common linear algebra functions

"""
import ufunc
import numpy
import ndarray

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
