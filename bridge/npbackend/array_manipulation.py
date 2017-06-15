"""
Array manipulation routines
===========================
"""
from . import array_create
import numpy_force as numpy
from . import bhary
from . import _util
from .bhary import fix_biclass_wrapper

@fix_biclass_wrapper
def flatten(ary, order='C', always_copy=True):
    """
    Return a copy of the array collapsed into one dimension.

    Parameters
    ----------
    ary : array_like
        Array from which to retrieve the flattened data from.
    order : {'C', 'F', 'A', 'K'}, optional
        'C' means to flatten in row-major (C-style) order.
        'F' means to flatten in column-major (Fortran-
        style) order. 'A' means to flatten in column-major
        order if `a` is Fortran *contiguous* in memory,
        row-major order otherwise. 'K' means to flatten
        `a` in the order the elements occur in memory.
        The default is 'C'.
    always_copy : boolean
        When False, a copy is only made when necessary

    Returns
    -------
    y : ndarray
        A copy of the input array, flattened to one dimension.

    Notes
    -----
    The order of the data in memory is always row-major (C-style).

    Examples
    --------
    >>> a = np.array([[1,2], [3,4]])
    >>> np.flatten(a)
    array([1, 2, 3, 4])
    """

    if order == 'F' or (order == 'A' and not ary.flags['F_CONTIGUOUS']):
        ary = numpy.transpose(ary)

    ret = ary.reshape(_util.totalsize(ary))
    if always_copy:
        return ret.copy()
    else:
        return ret


@fix_biclass_wrapper
def diagonal(ary, offset=0, axis1=0, axis2=1):
    """
    Return specified diagonals.

    If `a` is 2-D, returns the diagonal of `a` with the given offset,
    i.e., the collection of elements of the form ``a[i, i+offset]``.
    If `a` has more than two dimensions, then the axes specified by
    `axis1` and `axis2` are used to determine the 2-D sub-array whose
    diagonal is returned. The shape of the resulting array can be
    determined by removing `axis1` and `axis2` and appending an index
    to the right equal to the size of the resulting diagonals.

    Parameters
    ----------
    ary : array_like
        Array from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal.  Can be positive or
        negative.  Defaults to main diagonal (0).
    axis1 : int, optional
        Axis to be used as the first axis of the 2-D sub-arrays from which
        the diagonals should be taken. Defaults to first axis (0).
    axis2 : int, optional
        Axis to be used as the second axis of the 2-D sub-arrays from which
        the diagonals should be taken. Defaults to second axis (1).

    Returns
    -------
    array_of_diagonals : ndarray
        If `a` is 2-D, a 1-D array containing the diagonal is returned.
        If the dimension of `a` is larger, then an array of diagonals is
        returned, "packed" from left-most dimension to right-most (e.g.,
        if `a` is 3-D, then the diagonals are "packed" along rows).

    Raises
    ------
    ValueError
        If the dimension of `a` is less than 2.

    See Also
    --------
    diag : MATLAB work-a-like for 1-D and 2-D arrays.
    diagflat : Create diagonal arrays.
    trace : Sum along diagonals.

    Examples
    --------
    >>> a = np.arange(4).reshape(2,2); a
    array([[0, 1],
           [2, 3]])
    >>> a.diagonal()
    array([0, 3])
    >>> a.diagonal(1)
    array([1])

    A 3-D example:

    >>> a = np.arange(8).reshape(2,2,2); a
    array([[[0, 1],
            [2, 3]],

           [[4, 5],
            [6, 7]]])
    >>> a.diagonal()
    array([[0, 6],
           [1, 7]])
    """
    if axis1 == axis2:
        raise Exception("axis1 and axis2 cannot be the same\n")
    if ary.ndim < 2:
        raise Exception("diagonal requires an array of at least two dimensions\n")

    # Get all axes except the two which has the diagonal we seek;
    # these are added later
    min_axis, max_axis = sorted([axis1, axis2])
    tr = list(range(ary.ndim))
    del tr[max_axis]
    del tr[min_axis]

    # Positive offset means upper diagonals, negative is lower, so we switch
    # the axes around if negative
    if offset >= 0:
        ary = ary.transpose(tr + [axis1, axis2])
    else:
        ary = ary.transpose(tr + [axis2, axis1])
        offset = -offset

    # Calculate how many elements will be in the diagonal
    diag_size = max(0, min(ary.shape[-2], ary.shape[-1] - offset))
    ret_shape = ary.shape[:-2] + (diag_size,)

    # Return empty array if the diagonal has zero elements
    if diag_size == 0:
        return array_create.empty(ret_shape, dtype=ary.dtype, bohrium=bhary.check(ary))

    ary = ary[..., :diag_size, offset:(offset + diag_size)]

    ret_strides = ary.strides[:-2] + (ary.strides[-1] + ary.strides[-2],)
    return numpy.lib.stride_tricks.as_strided(ary, shape=ret_shape, strides=ret_strides)


@fix_biclass_wrapper
def diagflat(d, k=0):
    """
    Create a two-dimensional array with the flattened input as a diagonal.

    Parameters
    ----------
    d : array_like
        Input data, which is flattened and set as the `k`-th
        diagonal of the output.
    k : int, optional
        Diagonal to set; 0, the default, corresponds to the "main" diagonal,
        a positive (negative) `k` giving the number of the diagonal above
        (below) the main.

    Returns
    -------
    out : ndarray
        The 2-D output array.

    See Also
    --------
    diag : MATLAB work-alike for 1-D and 2-D arrays.
    diagonal : Return specified diagonals.
    trace : Sum along diagonals.

    Examples
    --------
    >>> np.diagflat([[1,2], [3,4]])
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])

    >>> np.diagflat([1,2], 1)
    array([[0, 1, 0],
           [0, 0, 2],
           [0, 0, 0]])

    """
    d = flatten(d)
    size = d.size+abs(k)
    A = array_create.zeros((size, size), dtype=d.dtype, bohrium=bhary.check(d))
    Ad = diagonal(A, offset=k)
    Ad[...] = d
    return A


@fix_biclass_wrapper
def diag(v, k=0):
    """
    Extract a diagonal or construct a diagonal array.

    Parameters
    ----------
    v : array_like
        If `v` is a 2-D array, return a copy of its `k`-th diagonal.
        If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
        diagonal.
    k : int, optional
        Diagonal in question. The default is 0. Use `k>0` for diagonals
        above the main diagonal, and `k<0` for diagonals below the main
        diagonal.

    Returns
    -------
    out : ndarray
        The extracted diagonal or constructed diagonal array.

    See Also
    --------
    diagonal : Return specified diagonals.
    diagflat : Create a 2-D array with the flattened input as a diagonal.
    trace : Sum along diagonals.
    triu : Upper triangle of an array.
    tril : Lower triange of an array.

    Examples
    --------
    >>> x = np.arange(9).reshape((3,3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    >>> np.diag(x)
    array([0, 4, 8])
    >>> np.diag(x, k=1)
    array([1, 5])
    >>> np.diag(x, k=-1)
    array([3, 7])

    >>> np.diag(np.diag(x))
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 8]])
    """

    if v.ndim == 1:
        return diagflat(v, k)
    elif v.ndim == 2:
        return diagonal(v, k)
    else:
        raise ValueError("Input must be 1- or 2-d.")


@fix_biclass_wrapper
def reshape(a, *newshape, **kwargs):
    """
    Gives a new shape to an array without changing its data.

    Parameters
    ----------
    a : array_like
        Array to be reshaped.
    newshape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is inferred
        from the length of the array and remaining dimensions.
    order : {`C`, `F`, `A`}, optional
        Read the elements of a using this index order, and place the elements
        into the reshaped array using this index order.
        `C` means to read / write the elements using C-like index order,
        with the last axis index changing fastest, back to the first axis
        index changing slowest. `F` means to read / write the elements using
        Fortran-like index order, with the first index changing fastest,
        and the last index changing slowest. Note that the `C` and `F` options
        take no account of the memory layout of the underlying array,
        and only refer to the order of indexing. `A` means to read / write
        the elements in Fortran-like index order if a is Fortran contiguous
        in memory, C-like order otherwise.

    Returns
    -------
    reshaped_array : ndarray
        This will be a new view object if possible; otherwise, it will
        be a copy.  Note there is no guarantee of the *memory layout* (C- or
        Fortran- contiguous) of the returned array.

    See Also
    --------
    ndarray.reshape : Equivalent method.

    Notes
    -----
    It is not always possible to change the shape of an array without
    copying the data. If you want an error to be raise if the data is copied,
    you should assign the new shape to the shape attribute of the array::

     >>> a = np.zeros((10, 2))
     # A transpose make the array non-contiguous
     >>> b = a.T
     # Taking a view makes it possible to modify the shape without modifying the
     # initial object.
     >>> c = b.view()
     >>> c.shape = (20)
     AttributeError: incompatible shape for a non-contiguous array

    The `order` keyword gives the index ordering both for *fetching* the values
    from `a`, and then *placing* the values into the output array.  For example,
    let's say you have an array:

    >>> a = np.arange(6).reshape((3, 2))
    >>> a
    array([[0, 1],
           [2, 3],
           [4, 5]])

    You can think of reshaping as first raveling the array (using the given
    index order), then inserting the elements from the raveled array into the
    new array using the same kind of index ordering as was used for the
    raveling.

    >>> np.reshape(a, (2, 3)) # C-like index ordering
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.reshape(np.ravel(a), (2, 3)) # equivalent to C ravel then C reshape
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.reshape(a, (2, 3), order='F') # Fortran-like index ordering
    array([[0, 4, 3],
           [2, 1, 5]])
    >>> np.reshape(np.ravel(a, order='F'), (2, 3), order='F')
    array([[0, 4, 3],
           [2, 1, 5]])

    Examples
    --------
    >>> a = np.array([[1,2,3], [4,5,6]])
    >>> np.reshape(a, 6)
    array([1, 2, 3, 4, 5, 6])
    >>> np.reshape(a, 6, order='F')
    array([1, 4, 2, 5, 3, 6])

    >>> np.reshape(a, (3,-1))       # the unspecified value is inferred to be 2
    array([[1, 2],
           [3, 4],
           [5, 6]])
    """
    # Let's make sure that newshape is a flat sequence
    if len(newshape) == 1:
        # The item is a sequence
        if hasattr(newshape[0], "__getitem__"):
            newshape = newshape[0]

    if not a.flags['C_CONTIGUOUS']:
        t = array_create.empty_like(a)
        t[...] = a
        a = t

    return numpy.ndarray.reshape(a, newshape, **kwargs)


@fix_biclass_wrapper
def trace(ary, offset=0, axis1=0, axis2=1, dtype=None):
    """
    Return the sum along diagonals of the array.

    If `a` is 2-D, the sum along its diagonal with the given offset
    is returned, i.e., the sum of elements ``a[i,i+offset]`` for all i.

    If `a` has more than two dimensions, then the axes specified by axis1 and
    axis2 are used to determine the 2-D sub-arrays whose traces are returned.
    The shape of the resulting array is the same as that of `a` with `axis1`
    and `axis2` removed.

    Parameters
    ----------
    ary : array_like
        Input array, from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal. Can be both positive
        and negative. Defaults to 0.
    axis1, axis2 : int, optional
        Axes to be used as the first and second axis of the 2-D sub-arrays
        from which the diagonals should be taken. Defaults are the first two
        axes of `a`.
    dtype : dtype, optional
        Determines the data-type of the returned array and of the accumulator
        where the elements are summed. If dtype has the value None and `a` is
        of integer type of precision less than the default integer
        precision, then the default integer precision is used. Otherwise,
        the precision is the same as that of `a`.
    out : ndarray, optional
        Array into which the output is placed. Its type is preserved and
        it must be of the right shape to hold the output.

    Returns
    -------
    sum_along_diagonals : ndarray
        If `a` is 2-D, the sum along the diagonal is returned.  If `a` has
        larger dimensions, then an array of sums along diagonals is returned.

    See Also
    --------
    diag, diagonal, diagflat

    Examples
    --------
    >>> np.trace(np.eye(3))
    3.0
    >>> a = np.arange(8).reshape((2,2,2))
    >>> np.trace(a)
    array([6, 8])

    >>> a = np.arange(24).reshape((2,2,2,3))
    >>> np.trace(a).shape
    (2, 3)

    """
    D = diagonal(ary, offset=offset, axis1=axis1, axis2=axis2)

    if dtype:
        D = D.astype(dtype)

    return D.sum(axis=-1)


@fix_biclass_wrapper
def broadcast_arrays(*args):
    """
    Broadcast any number of arrays against each other.

    .. note:: This function is very similar to NumPy's  `broadcast_arrays()`

    Parameters
    ----------
    `array_list` : array_likes
        The arrays to broadcast.

    Returns
    -------
    broadcasted : list of arrays
        These arrays are views on the original arrays or the untouched originals.
        They are typically not contiguous.  Furthermore, more than one element of a
        broadcasted array may refer to a single memory location.  If you
        need to write to the arrays, make copies first.
    shape : tuple
        The shape the arrays are broadcasted to
    """
    try:
        if len(args) == 0:
            return ([], [])

        # Common case where nothing needs to be broadcasted.
        bcast = numpy.broadcast(*args)
        if all(array.shape == bcast.shape for array in args if not numpy.isscalar(array)):    
            return (args, bcast.shape)

        ret = []
        # We use NumPy's broadcast_arrays() to broadcast the views.
        # Notice that the 'subok' argument is first introduced in version 10 of NumPy
        try:
            bargs = numpy.broadcast_arrays(*args, subok=True)
        except TypeError as err:
            if "subok" in err.message:
                bargs = numpy.broadcast_arrays(*args)
            else:
                raise

        for a, b in zip(args, bargs):
            if numpy.isscalar(a) or not isinstance(a, numpy.ndarray):
                ret.append(b)
            elif bhary.identical_views(a, b):
                ret.append(a)
            else:
                ret.append(b)
    except ValueError as msg:
        if str(msg).find("shape mismatch: objects cannot be broadcast to a single shape") != -1:
            shapes = [arg.shape for arg in args]
            raise ValueError("shape mismatch: objects cannot be broadcasted to a single shape: %s" % shapes)
        raise
    return (ret, bcast.shape)


@fix_biclass_wrapper
def fill(a, value):
    """
    a.fill(value)

    Fill the array with a scalar value.

    Parameters
    ----------
        a : array_like
        Array to fill
    value : scalar
        All elements of `a` will be assigned this value.

    Examples
    --------
    >>> a = np.array([1, 2])
    >>> a.fill(0)
    >>> a
    array([0, 0])
    >>> a = np.empty(2)
    >>> a.fill(1)
    >>> a
    array([ 1.,  1.])

    """
    a[...] = value


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
      >>> a = np.array([[1, 2], [3, 4]])
      >>> b = np.array([[5, 6]])
      >>> np.concatenate((a, b), axis=0)
      array([[1, 2],
             [3, 4],
             [5, 6]])
      >>> np.concatenate((a, b.T), axis=1)
      array([[1, 2, 5],
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
      array([2, 3, 4])
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
        exec(cmd)
        len_count += array_list[i].shape[axis]
    return ret
