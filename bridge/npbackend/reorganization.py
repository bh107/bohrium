"""
Reorganization of Array Elements Routines
===========================
"""
import warnings
import numpy_force as numpy
from . import bhary
from . import _util
from .bhary import fix_biclass_wrapper, get_bhc
from . import target
from . import array_create
from . import array_manipulation


@fix_biclass_wrapper
def gather(ary, indexes):
    """
    gather(ary, indexes)

    Gather elements from 'ary' selected by 'indexes'.
    The values of 'indexes' are absolute indexed into a flatten 'ary'
    The shape of the returned array equals indexes.shape.

    Parameters
    ----------
    ary  : array_like
        The array to gather elements from.
    indexes : array_like, interpreted as integers
        Array or list of indexes that will be gather from 'array'

    Returns
    -------
    r : ndarray
        The gathered array freshly-allocated.
    """

    ary = array_manipulation.flatten(array_create.array(ary))

    indexes = array_create.array(indexes, dtype=numpy.uint64, bohrium=True)
    ret = array_create.empty(indexes.shape, dtype=ary.dtype, bohrium=True)

    if ary.size == 0 or indexes.size == 0:
        return array_create.empty([])

    target.gather(get_bhc(ret), get_bhc(ary), get_bhc(indexes))
    return ret


@fix_biclass_wrapper
def take(a, indices, axis=None, out=None, mode='raise'):
    """
    Take elements from an array along an axis.

    This function does the same thing as "fancy" indexing (indexing arrays
    using arrays); however, it can be easier to use if you need elements
    along a given axis.

    Parameters
    ----------
    a : array_like
        The source array.
    indices : array_like, interpreted as integers
        The indices of the values to extract.

        .. versionadded:: 1.8.0

        Also allow scalars for indices.
    axis : int, optional
        The axis over which to select values. By default, the flattened
        input array is used.
    out : ndarray, optional
        If provided, the result will be placed in this array. It should
        be of the appropriate shape and dtype.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave.

        * 'raise' -- raise an error (default)
        * 'wrap' -- wrap around
        * 'clip' -- clip to the range

        'clip' mode means that all indices that are too large are replaced
        by the index that addresses the last element along that axis. Note
        that this disables indexing with negative numbers.

    Returns
    -------
    subarray : ndarray
        The returned array has the same type as `a`.

    See Also
    --------
    compress : Take elements using a boolean mask
    ndarray.take : equivalent method

    Examples
    --------
    >>> a = [4, 3, 5, 7, 6, 8]
    >>> indices = [0, 1, 4]
    >>> np.take(a, indices)
    array([4, 3, 6])

    In this example if `a` is an ndarray, "fancy" indexing can be used.

    >>> a = np.array(a)
    >>> a[indices]
    array([4, 3, 6])

    If `indices` is not one dimensional, the output also has these dimensions.

    >>> np.take(a, [[0, 1], [2, 3]])
    array([[4, 3],
           [5, 7]])
    """

    if not bhary.check(a):
        numpy.take(a, indices, axis=axis, out=out, mode=mode)

    if mode != "raise":
        warnings.warn("Bohrium only supports the 'raise' mode not '%s', "
                      "it will be handled by the original NumPy." % mode, UserWarning, 2)
        numpy.take(a, indices, axis=axis, out=out, mode=mode)

    if axis is not None and a.ndim > 1:
        warnings.warn("Bohrium does not support the 'axis' argument, "
                      "it will be handled by the original NumPy.", UserWarning, 2)
        numpy.take(a, indices, axis=axis, out=out, mode=mode)

    ret = gather(a, indices)
    if out is not None:
        out[...] = ret
        return out
    else:
        return ret


@fix_biclass_wrapper
def scatter(ary, indexes, values):
    """
    scatter(ary, indexes, values)

    Scatter 'values' into 'ary' selected by 'indexes'.
    The values of 'indexes' are absolute indexed into a flatten 'ary'
    The shape of 'indexes' and 'value' must be equal.

    Parameters
    ----------
    ary  : array_like
        The target array to write the values to.
    indexes : array_like, interpreted as integers
        Array or list of indexes that will be written to in 'ary'
    values : array_like
        Values to write into 'ary"
    """

    ary = array_create.array(ary)
    indexes = array_manipulation.flatten(array_create.array(indexes, dtype=numpy.uint64), always_copy=False)
    values = array_manipulation.flatten(array_create.array(values, dtype=ary.dtype), always_copy=False)

    assert indexes.shape == values.shape
    if ary.size == 0 or indexes.size == 0:
        return

    # In order to ensure a contiguous array, we do the scatter on a flatten copy
    flat = array_manipulation.flatten(ary, always_copy=True)
    target.scatter(get_bhc(flat), get_bhc(values), get_bhc(indexes))
    ary[...] = flat.reshape(ary.shape)


@fix_biclass_wrapper
def put(a, ind, v, mode='raise'):
    """
    Replaces specified elements of an array with given values.

    The indexing works on the flattened target array. `put` is roughly
    equivalent to:

    ::

        a.flat[ind] = v

    Parameters
    ----------
    a : ndarray
        Target array.
    ind : array_like
        Target indices, interpreted as integers.
    v : array_like
        Values to place in `a` at target indices. If `v` is shorter than
        `ind` it will be repeated as necessary.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave.

        * 'raise' -- raise an error (default)
        * 'wrap' -- wrap around
        * 'clip' -- clip to the range

        'clip' mode means that all indices that are too large are replaced
        by the index that addresses the last element along that axis. Note
        that this disables indexing with negative numbers.

    See Also
    --------
    putmask, place, take

    Examples
    --------
    >>> a = np.arange(5)
    >>> np.put(a, [0, 2], [-44, -55])
    >>> a
    array([-44,   1, -55,   3,   4])

    >>> a = np.arange(5)
    >>> np.put(a, 22, -5, mode='clip')
    >>> a
    array([ 0,  1,  2,  3, -5])

    """

    if not bhary.check(a):
        return numpy.put(a, ind, v, mode=mode)

    if mode != "raise":
        warnings.warn("Bohrium only supports the 'raise' mode not '%s', "
                      "it will be handled by the original NumPy." % mode, UserWarning, 2)
        return numpy.put(a, ind, v, mode=mode)

    if _util.totalsize(ind) != _util.totalsize(v):
        warnings.warn("Bohrium only supports 'ind' and 'v' having the same length, "
                      "it will be handled by the original NumPy.", UserWarning, 2)
        return numpy.put(a, ind, v, mode=mode)

    scatter(a, ind, v)


@fix_biclass_wrapper
def cond_scatter(ary, indexes, values, mask):
    """
    scatter(ary, indexes, values, mask)

    Scatter 'values' into 'ary' selected by 'indexes' where 'mask' is true.
    The values of 'indexes' are absolute indexed into a flatten 'ary'
    The shape of 'indexes', 'value', and 'mask' must be equal.


    Parameters
    ----------
    ary  : array_like
        The target array to write the values to.
    indexes : array_like, interpreted as integers
        Array or list of indexes that will be written to in 'ary'
    values : array_like
        Values to write into 'ary'
    mask : array_like, interpreted as booleans
        A mask that specifies which indexes and values to include and exclude
    """

    ary = array_create.array(ary)
    indexes = array_manipulation.flatten(array_create.array(indexes, dtype=numpy.uint64), always_copy=False)
    values = array_manipulation.flatten(array_create.array(values, dtype=ary.dtype), always_copy=False)
    mask = array_manipulation.flatten(array_create.array(mask, dtype=numpy.bool), always_copy=False)

    assert (indexes.shape == values.shape and values.shape == mask.shape)
    if ary.size == 0 or indexes.size == 0:
        return

    # In order to ensure a contiguous array, we do the scatter on a flatten copy
    flat = array_manipulation.flatten(ary, always_copy=True)
    target.cond_scatter(get_bhc(flat), get_bhc(values), get_bhc(indexes), get_bhc(mask))
    ary[...] = flat.reshape(ary.shape)
