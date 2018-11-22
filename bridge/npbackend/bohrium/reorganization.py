"""
Reorganization of Array Elements Routines
===========================
"""
import warnings
import numpy_force as numpy
from . import bhary
from bohrium_api import _info
from ._util import is_scalar
from .bhary import fix_biclass_wrapper
from . import array_create
from . import array_manipulation
from . import ufuncs
from . import numpy_backport


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
    from . import _bh

    ary = array_manipulation.flatten(array_create.array(ary))

    # Convert a scalar index to a 1-element array
    if is_scalar(indexes):
        indexes = [indexes]
        
    indexes = array_create.array(indexes, dtype=numpy.uint64, bohrium=True)
    ret = array_create.empty(indexes.shape, dtype=ary.dtype, bohrium=True)
    if ary.size == 0 or indexes.size == 0:
        return array_create.array([])

    _bh.ufunc(_info.op['gather']['id'], (ret, ary, indexes))
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
        indices = array_create.array(indices, bohrium=False)
        return numpy.take(a, indices, axis=axis, out=out, mode=mode)

    if mode != "raise":
        warnings.warn("Bohrium only supports the 'raise' mode not '%s', "
                      "it will be handled by the original NumPy." % mode, UserWarning, 2)
        a = array_create.array(a, bohrium=False)
        indices = array_create.array(indices, bohrium=False)
        return numpy.take(a, indices, axis=axis, out=out, mode=mode)

    if axis is not None and a.ndim > 1:
        warnings.warn("Bohrium does not support the 'axis' argument, "
                      "it will be handled by the original NumPy.", UserWarning, 2)
        a = array_create.array(a, bohrium=False)
        indices = array_create.array(indices, bohrium=False)
        return numpy.take(a, indices, axis=axis, out=out, mode=mode)

    ret = gather(a, indices)
    if out is not None:
        out[...] = ret
        return out
    else:
        return ret


@fix_biclass_wrapper
def take_using_index_tuple(a, index_tuple, out=None):
    """
    Take elements from the array 'a' specified by 'index_tuple'
    This function is very similar to take(), but takes a tuple of index arrays rather than a single index array
    
    Parameters
    ----------
    a : array_like
        The source array.
    index_tuple : tuple of array_like, interpreted as integers
        Each array in the tuple specified the indices of the values to extract for that axis. 
        The number of arrays in 'index_tuple' must equal the number of dimension in 'a'

    out : ndarray, optional
        If provided, the result will be placed in this array. It should
        be of the appropriate shape and dtype.


    Returns
    -------
    subarray : ndarray
        The returned array has the same type as `a`.

    """

    if not bhary.check(a):
        ret = a[index_tuple]
        if out is not None:
            out[...] = ret
            return out
        else:
            return ret

    assert len(index_tuple) == a.ndim

    if a.size == 0:
        return array_create.array([], dtype=a.dtype)

    if a.ndim == 1:
        return take(a, index_tuple[0], out=out)

    # Make sure that all index arrays are uint64 bohrium arrays
    index_list = []
    for index in index_tuple:
        index_list.append(array_create.array(index, dtype=numpy.uint64, bohrium=True))
        if index_list[-1].size == 0:
            return array_create.empty((0,), dtype=a.dtype)

    # And then broadcast them into the same shape
    index_list = array_manipulation.broadcast_arrays(*index_list)[0]

    # Let's find the absolute index
    abs_index = index_list[-1].copy()
    stride = a.shape[-1]
    for i in range(len(index_list) - 2, -1, -1):  # Iterate backwards from index_list[-2]
        abs_index += index_list[i] * stride
        stride *= a.shape[i]

    # take() support absolute indices
    ret = take(a, abs_index).reshape(index_list[0].shape)
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
    from . import _bh

    indexes = array_manipulation.flatten(array_create.array(indexes, dtype=numpy.uint64), always_copy=False)
    values = array_manipulation.flatten(array_create.array(values, dtype=ary.dtype), always_copy=False)

    assert indexes.shape == values.shape
    if ary.size == 0 or indexes.size == 0:
        return

    # In order to ensure a contiguous array, we do the scatter on a flatten copy
    flat = array_manipulation.flatten(ary, always_copy=True)
    _bh.ufunc(_info.op['scatter']['id'], (flat, values, indexes))
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

    if ind.size == 0:
        return  # Nothing to insert!

    if not bhary.check(a):
        return numpy.put(a, ind.astype(numpy.int64), v, mode=mode)

    if mode != "raise":
        warnings.warn("Bohrium only supports the 'raise' mode not '%s', "
                      "it will be handled by the original NumPy." % mode, UserWarning, 2)
        return numpy.put(a, ind, v, mode=mode)

    indexes = array_manipulation.flatten(array_create.array(ind, dtype=numpy.uint64), always_copy=False)
    values = array_manipulation.flatten(array_create.array(v, dtype=a.dtype), always_copy=False)

    # Now let's make the shape of 'indexes' and 'values' match
    if indexes.size > values.size:
        if values.size == 1:
            # When 'values' is a scalar, we can broadcast it to match 'indexes'
            values = numpy_backport.as_strided(values, shape=indexes.shape, strides=(0,))
        else:  # else we repeat 'values' enough times to be larger than 'indexes'
            values = numpy_backport.as_strided(values,
                                               shape=(indexes.size // values.size + 2, values.size),
                                               strides=(0, values.itemsize))
            values = array_manipulation.flatten(values, always_copy=False)

    # When 'values' is too large, we simple cut the end off
    if values.size > indexes.size:
        values = values[0:indexes.size]

    # Now that 'indexes' and 'values' have the same shape, we can call 'scatter'
    scatter(a, indexes, values)


@fix_biclass_wrapper
def put_using_index_tuple(a, index_tuple, v):
    """
    Replaces specified elements of an array with given values.
    This function is very similar to put(), but takes a tuple of index arrays rather than a single index array.
    The indexing works like fancy indexing:
    
    ::

        a[index_tuple] = v

    Parameters
    ----------
    a : array_like
        The source array.
    index_tuple : tuple of array_like, interpreted as integers
        Each array in the tuple specified the indices of the values to extract for that axis. 
        The number of arrays in 'index_tuple' must equal the number of dimension in 'a'
    v : array_like
        Values to place in `a`.

    Returns
    -------
    subarray : ndarray
        The returned array has the same type as `a`.
    """

    if not bhary.check(a):
        a[index_tuple] = array_create.array(v, bohrium=False)
        return

    v = array_create.array(v, bohrium=True)
    assert len(index_tuple) == a.ndim

    if a.size == 0:
        return

    if a.ndim == 1:
        return put(a, index_tuple[0], v)

    # Make sure that all index arrays are uint64 bohrium arrays
    index_list = []
    for index in index_tuple:
        index_list.append(array_create.array(index, dtype=numpy.uint64, bohrium=True))
        if index_list[-1].size == 0:
            return array_create.empty((0,), dtype=a.dtype)

    # And then broadcast them into the same shape
    index_list = array_manipulation.broadcast_arrays(*index_list)[0]

    # Let's find the absolute index
    abs_index = index_list[-1].copy()
    stride = a.shape[-1]
    for i in range(len(index_list) - 2, -1, -1):  # Iterate backwards from index_list[-2]
        abs_index += index_list[i] * stride
        stride *= a.shape[i]

    # put() support absolute indices
    put(a, abs_index, v)


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
    from . import _bh

    indexes = array_manipulation.flatten(array_create.array(indexes, dtype=numpy.uint64), always_copy=False)
    values = array_manipulation.flatten(array_create.array(values, dtype=ary.dtype), always_copy=False)
    mask = array_manipulation.flatten(array_create.array(mask, dtype=numpy.bool), always_copy=False)

    assert (indexes.shape == values.shape and values.shape == mask.shape)
    if ary.size == 0 or indexes.size == 0:
        return

    # In order to ensure a contiguous array, we do the scatter on a flatten copy
    flat = array_manipulation.flatten(ary, always_copy=True)
    _bh.ufunc(_info.op['cond_scatter']['id'], (flat, values, indexes, mask))
    ary[...] = flat.reshape(ary.shape)


@fix_biclass_wrapper
def pack(ary, mask):
    """
    pack(ary, mask)

    Packing the elements of 'ary' specified by 'mask' into new array that are contiguous
    The values of 'indexes' are absolute indexed into a flatten 'ary'
    The shape of 'mask' and 'ary' must be equal.


    Parameters
    ----------
    ary  : array_like, read flatten
        The array to read from.
    mask : array_like, interpreted as a flatten boolean array
        A mask that specifies which indexes of 'ary' to read
    """

    ary = array_manipulation.flatten(array_create.array(ary), always_copy=False)
    mask = array_manipulation.flatten(array_create.array(mask, dtype=numpy.bool), always_copy=False)
    assert (ary.shape == mask.shape)
    if ary.size == 0 or mask.size == 0:
        return

    true_indexes = ufuncs.add.accumulate(mask)
    true_count = int(true_indexes[-1])
    if true_count == 0:
        return array_create.empty((0,), dtype=ary.dtype)
    else:
        ret = array_create.empty((true_count,), dtype=ary.dtype)
        cond_scatter(ret, true_indexes - 1, ary, mask)
        return ret


@fix_biclass_wrapper
def flatnonzero(a):
    """
    Return indices that are non-zero in the flattened version of a.
    This is equivalent to a.ravel().nonzero()[0].
    Parameters
    ----------
    a : ndarray
        Input array.
    Returns
    -------
    res : ndarray
        Output array, containing the indices of the elements of `a.ravel()`
        that are non-zero.
    See Also
    --------
    nonzero : Return the indices of the non-zero elements of the input array.
    ravel : Return a 1-D array containing the elements of the input array.
    Examples
    --------
    >>> x = np.arange(-2, 3)
    >>> x
    array([-2, -1,  0,  1,  2])
    >>> np.flatnonzero(x)
    array([0, 1, 3, 4])
    Use the indices of the non-zero elements as an index array to extract
    these elements:
    >>> x.ravel()[np.flatnonzero(x)]
    array([-2, -1,  1,  2])
    """

    if a.dtype is not numpy.bool:
        mask = a != 0
    new_indexes = array_create.arange(a.size, dtype=numpy.uint64)
    return pack(new_indexes, mask)


@fix_biclass_wrapper
def nonzero(a):
    """
    Return the indices of the elements that are non-zero.
    Returns a tuple of arrays, one for each dimension of `a`,
    containing the indices of the non-zero elements in that
    dimension. The values in `a` are always tested and returned in
    row-major, C-style order. The corresponding non-zero
    values can be obtained with::
        a[nonzero(a)]
    To group the indices by element, rather than dimension, use::
        transpose(nonzero(a))
    The result of this is always a 2-D array, with a row for
    each non-zero element.
    Parameters
    ----------
    a : array_like
        Input array.
    Returns
    -------
    tuple_of_arrays : tuple
        Indices of elements that are non-zero.
    See Also
    --------
    flatnonzero :
        Return indices that are non-zero in the flattened version of the input
        array.
    ndarray.nonzero :
        Equivalent ndarray method.
    count_nonzero :
        Counts the number of non-zero elements in the input array.
    Examples
    --------
    >>> x = np.eye(3)
    >>> x
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> np.nonzero(x)
    (array([0, 1, 2]), array([0, 1, 2]))
    >>> x[np.nonzero(x)]
    array([ 1.,  1.,  1.])
    >>> np.transpose(np.nonzero(x))
    array([[0, 0],
           [1, 1],
           [2, 2]])
    A common use for ``nonzero`` is to find the indices of an array, where
    a condition is True.  Given an array `a`, the condition `a` > 3 is a
    boolean array and since False is interpreted as 0, np.nonzero(a > 3)
    yields the indices of the `a` where the condition is true.
    >>> a = np.array([[1,2,3],[4,5,6],[7,8,9]])
    >>> a > 3
    array([[False, False, False],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)
    >>> np.nonzero(a > 3)
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))
    The ``nonzero`` method of the boolean array can also be called.
    >>> (a > 3).nonzero()
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))
    """

    if a.ndim == 1:
        return (flatnonzero(a),)

    if not a.flags['C_CONTIGUOUS']:
        a = a.copy(order='C')

    nz = flatnonzero(a)
    ret = []
    for stride_in_bytes in a.strides:
        stride = stride_in_bytes // a.itemsize
        assert stride_in_bytes % a.itemsize == 0
        tmp = nz // stride
        ret.append(tmp)
        nz -= tmp * stride
    return ret
