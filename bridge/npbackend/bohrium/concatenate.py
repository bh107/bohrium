"""
Array concatenate functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
from . import array_create


def atleast_1d(*arys):
    """
    Convert inputs to arrays with at least one dimension.

    Scalar inputs are converted to 1-dimensional arrays, whilst
    higher-dimensional inputs are preserved.

    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more input arrays.

    Returns
    -------
    ret : ndarray
        An array, or list of arrays, each with ``a.ndim >= 1``.
        Copies are made only if necessary.

    See Also
    --------
    atleast_2d, atleast_3d

    Examples
    --------
    >>> np.atleast_1d(1.0)
    array_create.array([ 1.])

    >>> x = np.arange(9.0).reshape(3,3)
    >>> np.atleast_1d(x)
    array_create.array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7.,  8.]])
    >>> np.atleast_1d(x) is x
    True

    >>> np.atleast_1d(1, [3, 4])
    [array_create.array([1]), array_create.array([3, 4])]

    """
    res = []
    for ary in arys:
        ary = array_create.array(ary)
        if len(ary.shape) == 0:
            result = ary.reshape(1)
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res


def atleast_2d(*arys):
    """
    View inputs as arrays with at least two dimensions.

    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more array-like sequences.  Non-array inputs are converted
        to arrays.  Arrays that already have two or more dimensions are
        preserved.

    Returns
    -------
    res, res2, ... : ndarray
        An array, or list of arrays, each with ``a.ndim >= 2``.
        Copies are avoided where possible, and views with two or more
        dimensions are returned.

    See Also
    --------
    atleast_1d, atleast_3d

    Examples
    --------
    >>> np.atleast_2d(3.0)
    array_create.array([[ 3.]])

    >>> x = np.arange(3.0)
    >>> np.atleast_2d(x)
    array_create.array([[ 0.,  1.,  2.]])
    >>> np.atleast_2d(x).base is x
    True

    >>> np.atleast_2d(1, [1, 2], [[1, 2]])
    [array_create.array([[1]]), array_create.array([[1, 2]]), array_create.array([[1, 2]])]

    """
    res = []
    for ary in arys:
        ary = array_create.array(ary)
        if len(ary.shape) == 0:
            result = ary.reshape(1, 1)
        elif len(ary.shape) == 1:
            result = ary[None, :]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res


def atleast_3d(*arys):
    """
    View inputs as arrays with at least three dimensions.

    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more array-like sequences.  Non-array inputs are converted to
        arrays.  Arrays that already have three or more dimensions are
        preserved.

    Returns
    -------
    res1, res2, ... : ndarray
        An array, or list of arrays, each with ``a.ndim >= 3``.  Copies are
        avoided where possible, and views with three or more dimensions are
        returned.  For example, a 1-D array of shape ``(N,)`` becomes a view
        of shape ``(1, N, 1)``, and a 2-D array of shape ``(M, N)`` becomes a
        view of shape ``(M, N, 1)``.

    See Also
    --------
    atleast_1d, atleast_2d

    Examples
    --------
    >>> np.atleast_3d(3.0)
    array_create.array([[[ 3.]]])

    >>> x = np.arange(3.0)
    >>> np.atleast_3d(x).shape
    (1, 3, 1)

    >>> x = np.arange(12.0).reshape(4,3)
    >>> np.atleast_3d(x).shape
    (4, 3, 1)
    >>> np.atleast_3d(x).base is x.base  # x is a reshape, so not base itself
    True

    >>> for arr in np.atleast_3d([1, 2], [[1, 2]], [[[1, 2]]]):
    ...     print(arr, arr.shape)
    ...
    [[[1]
      [2]]] (1, 2, 1)
    [[[1]
      [2]]] (1, 2, 1)
    [[[1 2]]] (1, 1, 2)

    """
    res = []
    for ary in arys:
        ary = array_create.array(ary)
        if len(ary.shape) == 0:
            result = ary.reshape(1, 1, 1)
        elif len(ary.shape) == 1:
            result = ary[None, :, None]
        elif len(ary.shape) == 2:
            result = ary[:, :, None]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res


def concatenate(array_list, axis=0):
    """
      concatenate((a1, a2, ...), axis=0)

      Join a sequence of arrays along an existing axis.

      Parameters
      ----------
      a1, a2, ... : sequence of array_like
          The arrays must have the same shape, except in the dimension
          corresponding to `axis` (the first, by default).
      axis : int, optional
          The axis along which the arrays will be joined.  Default is 0.

      Returns
      -------
      res : ndarray
          The concatenated array.

      See Also
      --------
      ma.concatenate : Concatenate function that preserves input masks.
      array_split : Split an array into multiple sub-arrays of equal or
                    near-equal size.
      split : Split array into a list of multiple sub-arrays of equal size.
      hsplit : Split array into multiple sub-arrays horizontally (column wise)
      vsplit : Split array into multiple sub-arrays vertically (row wise)
      dsplit : Split array into multiple sub-arrays along the 3rd axis (depth).
      stack : Stack a sequence of arrays along a new axis.
      hstack : Stack arrays in sequence horizontally (column wise)
      vstack : Stack arrays in sequence vertically (row wise)
      dstack : Stack arrays in sequence depth wise (along third dimension)

      Notes
      -----
      When one or more of the arrays to be concatenated is a MaskedArray,
      this function will return a MaskedArray object instead of an ndarray,
      but the input masks are *not* preserved. In cases where a MaskedArray
      is expected as input, use the ma.concatenate function from the masked
      array module instead.

      Examples
      --------
      >>> a = np.array_create.array([[1, 2], [3, 4]])
      >>> b = np.array_create.array([[5, 6]])
      >>> np.concatenate((a, b), axis=0)
      array_create.array([[1, 2],
             [3, 4],
             [5, 6]])
      >>> np.concatenate((a, b.T), axis=1)
      array_create.array([[1, 2, 5],
             [3, 4, 6]])

      This function will not preserve masking of MaskedArray inputs.

      >>> a = np.ma.arange(3)
      >>> a[1] = np.ma.masked
      >>> b = np.arange(2, 5)
      >>> a
      masked_array(data = [0 -- 2],
                   mask = [False  True False],
             fill_value = 999999)
      >>> b
      array_create.array([2, 3, 4])
      >>> np.concatenate([a, b])
      masked_array(data = [0 1 2 2 3 4],
                   mask = False,
             fill_value = 999999)
      >>> np.ma.concatenate([a, b])
      masked_array(data = [0 -- 2 2 3 4],
                   mask = [False  True False False False False],
             fill_value = 999999)

    """

    if len(array_list) == 0:
        return None

    # We form an assignment to the new 'ret' array, which has a shape[axis] that are the sum of
    # the axis dimensions in 'array_list'. Then we copy each array in 'array_list' into the axis dimension of 'ret'

    ret_shape = list(array_list[0].shape)
    ret_shape[axis] = 0
    for ary in array_list:
        ret_shape[axis] += ary.shape[axis]
    ret = array_create.empty(ret_shape, dtype=array_list[0].dtype)

    slice = "ret["
    for i in range(ret.ndim):
        if i == axis:
            slice += "AXIS"
        else:
            slice += ":"
        if i < ret.ndim - 1:
            slice += ", "
    slice += "]"

    len_count = 0
    for i in range(len(array_list)):
        axis_slice = "%d:%d+%d" % (len_count, len_count, array_list[i].shape[axis])
        cmd = slice.replace("AXIS", axis_slice)
        cmd += " = array_list[i]"
        exec (cmd)
        len_count += array_list[i].shape[axis]
    return ret


def vstack(tup):
    """
    Stack arrays in sequence vertically (row wise).

    Take a sequence of arrays and stack them vertically to make a single
    array. Rebuild arrays divided by `vsplit`.

    This function continues to be supported for backward compatibility, but
    you should prefer ``np.concatenate`` or ``np.stack``. The ``np.stack``
    function was added in NumPy 1.10.

    Parameters
    ----------
    tup : sequence of ndarrays
        Tuple containing arrays to be stacked. The arrays must have the same
        shape along all but the first axis.

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays.

    See Also
    --------
    stack : Join a sequence of arrays along a new axis.
    hstack : Stack arrays in sequence horizontally (column wise).
    dstack : Stack arrays in sequence depth wise (along third dimension).
    concatenate : Join a sequence of arrays along an existing axis.
    vsplit : Split array into a list of multiple sub-arrays vertically.

    Notes
    -----
    Equivalent to ``np.concatenate(tup, axis=0)`` if `tup` contains arrays that
    are at least 2-dimensional.

    Examples
    --------
    >>> a = np.array_create.array([1, 2, 3])
    >>> b = np.array_create.array([2, 3, 4])
    >>> np.vstack((a,b))
    array_create.array([[1, 2, 3],
           [2, 3, 4]])

    >>> a = np.array_create.array([[1], [2], [3]])
    >>> b = np.array_create.array([[2], [3], [4]])
    >>> np.vstack((a,b))
    array_create.array([[1],
           [2],
           [3],
           [2],
           [3],
           [4]])

    """
    return concatenate([atleast_2d(_m) for _m in tup], 0)


def hstack(tup):
    """
    Stack arrays in sequence horizontally (column wise).

    Take a sequence of arrays and stack them horizontally to make
    a single array. Rebuild arrays divided by `hsplit`.

    This function continues to be supported for backward compatibility, but
    you should prefer ``np.concatenate`` or ``np.stack``. The ``np.stack``
    function was added in NumPy 1.10.

    Parameters
    ----------
    tup : sequence of ndarrays
        All arrays must have the same shape along all but the second axis.

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays.

    See Also
    --------
    stack : Join a sequence of arrays along a new axis.
    vstack : Stack arrays in sequence vertically (row wise).
    dstack : Stack arrays in sequence depth wise (along third axis).
    concatenate : Join a sequence of arrays along an existing axis.
    hsplit : Split array along second axis.

    Notes
    -----
    Equivalent to ``np.concatenate(tup, axis=1)``

    Examples
    --------
    >>> a = np.array((1,2,3))
    >>> b = np.array((2,3,4))
    >>> np.hstack((a,b))
    array_create.array([1, 2, 3, 2, 3, 4])
    >>> a = np.array_create.array([[1],[2],[3]])
    >>> b = np.array_create.array([[2],[3],[4]])
    >>> np.hstack((a,b))
    array_create.array([[1, 2],
           [2, 3],
           [3, 4]])

    """
    arrs = [atleast_1d(_m) for _m in tup]
    # As a special case, dimension 0 of 1-dimensional arrays is "horizontal"
    if arrs[0].ndim == 1:
        return concatenate(arrs, 0)
    else:
        return concatenate(arrs, 1)


def stack(arrays, axis=0):
    """
    Join a sequence of arrays along a new axis.

    The `axis` parameter specifies the index of the new axis in the dimensions
    of the result. For example, if ``axis=0`` it will be the first dimension
    and if ``axis=-1`` it will be the last dimension.

    .. versionadded:: 1.10.0

    Parameters
    ----------
    arrays : sequence of array_like
        Each array must have the same shape.
    axis : int, optional
        The axis in the result array along which the input arrays are stacked.

    Returns
    -------
    stacked : ndarray
        The stacked array has one more dimension than the input arrays.

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    split : Split array into a list of multiple sub-arrays of equal size.

    Examples
    --------
    >>> arrays = [np.random.randn(3, 4) for _ in range(10)]
    >>> np.stack(arrays, axis=0).shape
    (10, 3, 4)

    >>> np.stack(arrays, axis=1).shape
    (3, 10, 4)

    >>> np.stack(arrays, axis=2).shape
    (3, 4, 10)

    >>> a = np.array_create.array([1, 2, 3])
    >>> b = np.array_create.array([2, 3, 4])
    >>> np.stack((a, b))
    array_create.array([[1, 2, 3],
           [2, 3, 4]])

    >>> np.stack((a, b), axis=-1)
    array_create.array([[1, 2],
           [2, 3],
           [3, 4]])

    """
    arrays = [array_create.array(arr) for arr in arrays]
    if not arrays:
        raise ValueError('need at least one array to stack')

    shapes = set(arr.shape for arr in arrays)
    if len(shapes) != 1:
        raise ValueError('all input arrays must have the same shape')

    result_ndim = arrays[0].ndim + 1
    if not -result_ndim <= axis < result_ndim:
        msg = 'axis {0} out of bounds [-{1}, {1})'.format(axis, result_ndim)
        raise IndexError(msg)
    if axis < 0:
        axis += result_ndim

    sl = (slice(None),) * axis + (None,)
    expanded_arrays = [arr[sl] for arr in arrays]
    return concatenate(expanded_arrays, axis=axis)
