import numpy as np

from .bharray import implements


@implements(np.copy)
def copy(arr):
    """Return a copy of the array.

    Returns
    -------
    out : BhArray
        Copy of `arr`
    """
    return arr.copy()


@implements(np.transpose)
def transpose(arr, axes=None):
    """Permute the dimensions of an array.

    Parameters
    ----------
    axes : list of ints, optional
        By default, reverse the dimensions, otherwise permute the axes
        according to the values given.
    """
    if axes is None:
        axes = list(reversed(range(len(arr._shape))))

    ret = arr.view()
    ret._shape = tuple([arr._shape[i] for i in axes])
    ret._strides = tuple([arr._strides[i] for i in axes])
    return ret


@implements(np.ravel)
def ravel(arr):
    """ Return a contiguous flattened array.

    A 1-D array, containing the elements of the input, is returned. A copy is made only if needed.

    Returns
    -------
    y : ndarray
        A copy or view of the array, flattened to one dimension.
    """
    return arr.flatten(always_copy=False)


@implements(np.reshape)
def reshape(arr, shape):
    return arr.reshape(shape)


@implements(np.mean)
def mean(a, axis=None, dtype=None, out=None):
    import warnings
    from .ufuncs import ufunc_dict

    add = ufunc_dict['add']

    def _count_reduce_items(arr, axis):
        if axis is None:
            axis = tuple(range(arr.ndim))
        if not isinstance(axis, tuple):
            axis = (axis,)
        items = 1
        for ax in axis:
            items *= arr.shape[ax]
        return items

    def _mean(arr, axis=None, dtype=None, out=None):
        rcount = _count_reduce_items(arr, axis)
        # Make this warning show up first
        if rcount == 0:
            warnings.warn("Mean of empty slice.", RuntimeWarning, stacklevel=2)

        # Cast bool, unsigned int, and int to float64 by default
        if dtype is None:
            if issubclass(arr.dtype, (np.integer, np.bool_)):
                dtype = np.dtype('f8')

        ret = add.reduce(arr, axis=axis, dtype=dtype, out=out)
        if ret.isscalar():
            ret = ret.dtype(ret)
        ret /= rcount
        return ret

    return _mean(a, axis=axis, dtype=dtype, out=out)
