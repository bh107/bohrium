"""
Reorganization of Array Elements Routines
===========================
"""
import warnings
import numpy_force as numpy
from . import bhary
from .bhary import fix_biclass_wrapper, get_bhc
from . import ufuncs
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
    indexes : array_like
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
    indices : array_like
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