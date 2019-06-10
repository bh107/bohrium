import numpy as np
from . import bharray, ufuncs, array_create
from bohrium_api import _bh_api, _info


def gather(ary, indexes):
    """Gather elements from 'ary' selected by 'indexes'.

    The values of 'indexes' are absolute indexed into a flatten 'ary'
    The shape of the returned array equals indexes.shape.

    Parameters
    ----------
    ary  : BhArray
        The array to gather elements from.
    indexes : array_like, interpreted as integers
        Array or list of indexes that will be gather from 'array'

    Returns
    -------
    r : BhArray
        The gathered array freshly-allocated.
    """

    # NB: The code cache in Bohrium doesn't support views in GATHER.
    #     This could be fixed but it is more efficient to do a copy.
    ary = ary.flatten(always_copy=not ary.isbehaving())

    # Convert a scalar index to a 1-element array
    if np.isscalar(indexes):
        indexes = [indexes]

    # Make sure that indexes is BhArray of type `uint64`
    indexes = array_create.array(indexes, dtype=np.uint64, copy=False)

    if ary.nelem == 0 or indexes.nelem == 0:
        return bharray.BhArray(shape=0, dtype=ary.dtype)

    ret = bharray.BhArray(indexes.shape, dtype=ary.dtype)

    # BH_GATHER: Gather elements from IN selected by INDEX into OUT. NB: OUT.shape == INDEX.shape
    #            and IN can have any shape but must be contiguous.
    #            gather(OUT, IN, INDEX)
    ufuncs._call_bh_api_op(_info.op['gather']['id'], ret, (ary, indexes), broadcast_to_output_shape=False)
    return ret


def take(a, indices, axis=None, mode='raise'):
    """Take elements from an array along an axis.

    This function does the same thing as "fancy" indexing (indexing arrays
    using arrays); however, it can be easier to use if you need elements
    along a given axis.

    Parameters
    ----------
    a : array_like
        The source array.
    indices : array_like, interpreted as integers
        The indices of the values to extract.
        Also allow scalars for indices.
    axis : int, optional
        The axis over which to select values. By default, the flattened
        input array is used.
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
    r : BhArray
        The returned array has the same type as `a`.
    """

    a = array_create.array(a)

    if mode != "raise":
        raise NotImplementedError("Bohrium only supports the 'raise' mode not '%s'" % mode)

    if axis is not None and a.ndim > 1:
        raise NotImplementedError("Bohrium does not support the 'axis' argument")

    return gather(a, indices)


def take_using_index_tuple(a, index_tuple):
    """Take elements from the array 'a' specified by 'index_tuple'
    This function is very similar to take(), but takes a tuple of index arrays rather than a single index array

    Parameters
    ----------
    a : array_like
        The source array.
    index_tuple : tuple of array_like, interpreted as integers
        Each array in the tuple specified the indices of the values to extract for that axis.
        The number of arrays in 'index_tuple' must equal the number of dimension in 'a'

    Returns
    -------
    r : BhArray
        The returned array has the same type as `a`.
    """
    a = array_create.array(a)

    if len(index_tuple) != a.ndim:
        raise ValueError("length of `index_tuple` must equal the number of dimension in `a`")

    if a.size == 0:
        return array_create.empty(tuple(), dtype=a.dtype)

    if a.ndim == 1:
        return take(a, index_tuple[0])

    # Make sure that all index arrays are uint64 bohrium arrays
    index_list = []
    for index in index_tuple:
        index_list.append(array_create.array(index, dtype=np.uint64))
        if index_list[-1].size == 0:
            return array_create.empty(tuple(), dtype=a.dtype)

    # And then broadcast them into the same shape
    index_list = ufuncs.broadcast_arrays(index_list)

    # Let's find the absolute index
    abs_index = index_list[-1].copy()
    stride = a.shape[-1]
    for i in range(len(index_list) - 2, -1, -1):  # Iterate backwards from index_list[-2]
        abs_index += index_list[i] * stride
        stride *= a.shape[i]

    # take() support absolute indices
    return take(a, abs_index).reshape(index_list[0].shape)


def scatter(ary, indexes, values):
    """Scatter 'values' into 'ary' selected by 'indexes'.
    The values of 'indexes' are absolute indexed into a flatten 'ary'
    The shape of 'indexes' and 'value' must be equal.

    Parameters
    ----------
    ary  : BhArray
        The target array to write the values to.
    indexes : array_like, interpreted as integers
        Array or list of indexes that will be written to in 'ary'
    values : array_like
        Values to write into 'ary"
    """

    # Make sure that indexes is BhArray of type `uint64` and flatten
    indexes = array_create.array(indexes, dtype=np.uint64).flatten(always_copy=False)
    values = array_create.array(values, dtype=ary.dtype).flatten(always_copy=False)

    assert indexes.shape == values.shape
    if ary.size == 0 or indexes.size == 0:
        return

    # In order to ensure a contiguous array, we do the scatter on a flatten copy
    flat = ary.flatten(always_copy=True)

    # BH_SCATTER: Scatter all elements of IN into OUT selected by INDEX. NB: IN.shape == INDEX.shape
    #             and OUT can have any shape but must be contiguous.
    #             scatter(OUT, IN, INDEX)
    ufuncs._call_bh_api_op(_info.op['scatter']['id'], flat, (values, indexes), broadcast_to_output_shape=False)
    ary[...] = flat.reshape(ary.shape)


def cond_scatter(ary, indexes, values, mask):
    """ Scatter 'values' into 'ary' selected by 'indexes' where 'mask' is true.
    The values of 'indexes' are absolute indexed into a flatten 'ary'
    The shape of 'indexes', 'value', and 'mask' must be equal.


    Parameters
    ----------
    ary  : BhArray
        The target array to write the values to.
    indexes : array_like, interpreted as integers
        Array or list of indexes that will be written to in 'ary'
    values : array_like
        Values to write into 'ary'
    mask : array_like, interpreted as booleans
        A mask that specifies which indexes and values to include and exclude
    """
    indexes = array_create.array(indexes, dtype=np.uint64).flatten(always_copy=False)
    values = array_create.array(values, dtype=ary.dtype).flatten(always_copy=False)
    mask = array_create.array(mask, dtype=np.bool).flatten(always_copy=False)
    assert (indexes.shape == values.shape and values.shape == mask.shape)
    if ary.size == 0 or indexes.size == 0:
        return

    # In order to ensure a contiguous array, we do the scatter on a flatten copy
    flat = ary.flatten(always_copy=True)

    # BH_COND_SCATTER: Conditional scatter elements of IN where COND is true into OUT selected by INDEX.
    #                  NB: IN.shape == INDEX.shape and OUT can have any shape but must be contiguous.
    #                  cond_scatter(OUT, IN, INDEX, COND)
    ufuncs._call_bh_api_op(_info.op['cond_scatter']['id'], flat, (values, indexes, mask),
                           broadcast_to_output_shape=False)
    ary[...] = flat.reshape(ary.shape)


def put(a, ind, v, mode='raise'):
    """Replaces specified elements of an array with given values.

    The indexing works on the flattened target array. `put` is roughly
    equivalent to:

        a.flat[ind] = v

    Parameters
    ----------
    a : BhArray
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
    """

    if ind.size == 0:
        return  # Nothing to insert!

    if mode != "raise":
        raise NotImplementedError("Bohrium only supports the 'raise' mode not '%s'" % mode)

    # Make sure that indexes is BhArray of type `uint64` and flatten
    indexes = array_create.array(ind, dtype=np.uint64).flatten(always_copy=False)
    values = array_create.array(v, dtype=a.dtype).flatten(always_copy=False)

    # Now let's make the shape of 'indexes' and 'values' match
    if indexes.size > values.size:
        if values.size == 1:
            # When 'values' is a scalar, we can broadcast it to match 'indexes'
            values = bharray.BhArray(indexes.shape, values.dtype, strides=(0,), base=values.base, offset=values.offset)
        else:  # else we repeat 'values' enough times to be larger than 'indexes'
            values = bharray.BhArray((indexes.size // values.size + 2, values.size), values.dtype, strides=(0, 1),
                                     base=values.base, offset=values.offset)
            values = values.flatten(always_copy=False)

    # When 'values' is too large, we simple cut the end off
    if values.size > indexes.size:
        values = values[0:indexes.size]

    # Now that 'indexes' and 'values' have the same shape, we can call 'scatter'
    scatter(a, indexes, values)


def put_using_index_tuple(a, index_tuple, v):
    """Replaces specified elements of an array with given values.
    This function is very similar to put(), but takes a tuple of index arrays rather than a single index array.
    The indexing works like fancy indexing:

        a[index_tuple] = v

    Parameters
    ----------
    a : BhArray
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

    v = array_create.array(v)
    assert len(index_tuple) == a.ndim

    if a.size == 0:
        return

    if a.ndim == 1:
        return put(a, index_tuple[0], v)

    # Make sure that all index arrays are uint64 bohrium arrays
    index_list = []
    for index in index_tuple:
        index_list.append(array_create.array(index, dtype=np.uint64))
        if index_list[-1].size == 0:
            return array_create.empty(tuple(), dtype=a.dtype)

    # And then broadcast them into the same shape
    index_list = ufuncs.broadcast_arrays(index_list)

    # Let's find the absolute index
    abs_index = index_list[-1].copy()
    stride = a.shape[-1]
    for i in range(len(index_list) - 2, -1, -1):  # Iterate backwards from index_list[-2]
        abs_index += index_list[i] * stride
        stride *= a.shape[i]

    # put() support absolute indices
    put(a, abs_index, v)
