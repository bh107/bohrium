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
    in_operands = (ary, indexes)
    ufuncs._call_bh_api_op(_info.op['gather']['id'], ret, in_operands, broadcast_to_output_shape=False)
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

    if mode != "raise":
        raise NotImplementedError("Bohrium only supports the 'raise' mode not '%s'")

    if axis is not None and a.ndim > 1:
        raise NotImplementedError("Bohrium does not support the 'axis' argument")

    return gather(a, indices)
