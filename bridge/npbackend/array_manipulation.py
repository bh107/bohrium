"""
Array manipulation routines
===========================

"""
from . import array_create
import numpy_force as numpy
from . import bhary
from .bhary import fix_returned_biclass
import itertools

@fix_returned_biclass
def flatten(ary):
    """
    Return a copy of the array collapsed into one dimension.

    Parameters
    ----------
    a : array_like
        Array from which to retrive the flattened data from.

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
    return ary.reshape(numpy.multiply.reduce(numpy.asarray(ary.shape)))

@fix_returned_biclass
def trace(ary, offset=0, axis1=0, axis2=1, dtype=None):
    D = diagonal(ary, offset=offset, axis1=axis1, axis2=axis2)
    if dtype:
        D = D.astype(dtype)
    return numpy.add.reduce(D, axis=D.ndim-1)

@fix_returned_biclass
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
        return numpy.empty(ret_shape, dtype=ary.dtype)

    ary = ary[..., :diag_size, offset:(offset + diag_size)]

    ret_strides = ary.strides[:-2] + (ary.strides[-1] + ary.strides[-2],)
    return numpy.lib.stride_tricks.as_strided(ary, shape=ret_shape, strides=ret_strides)

@fix_returned_biclass
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

@fix_returned_biclass
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

@fix_returned_biclass
def reshape(a, *newshape):
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
    #Lets make sure that newshape is a flat sequence
    if len(newshape) == 1:
        if hasattr(newshape[0], "__getitem__"):#The item is a sequence
            newshape = newshape[0]

    if not a.flags['C_CONTIGUOUS']:
        t = array_create.empty_like(a)
        t[...] = a
        a = t
    return numpy.ndarray.reshape(a, newshape)

def broadcast_arrays(*args):
    """
    Broadcast any number of arrays against each other.

    .. note:: This function differ from NumPy in one way: it does not touch arrays that does not need broadcasting

    Parameters
    ----------
    `*args` : array_likes
        The arrays to broadcast.

    Returns
    -------
    broadcasted : list of arrays
        These arrays are views on the original arrays or the untouched originals.
        They are typically not contiguous.  Furthermore, more than one element of a
        broadcasted array may refer to a single memory location.  If you
        need to write to the arrays, make copies first.

    Examples
    --------
    >>> x = np.array([[1,2,3]])
    >>> y = np.array([[1],[2],[3]])
    >>> np.broadcast_arrays(x, y)
    [array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]]), array([[1, 1, 1],
           [2, 2, 2],
           [3, 3, 3]])]

    Here is a useful idiom for getting contiguous copies instead of
    non-contiguous views.

    >>> [np.array(a) for a in np.broadcast_arrays(x, y)]
    [array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]]), array([[1, 1, 1],
           [2, 2, 2],
           [3, 3, 3]])]

    """
    ret = []
    bargs = numpy.broadcast_arrays(*args)
    for a, b in itertools.izip(args, bargs):
        if numpy.isscalar(a):
            ret.append(b)
        elif bhary.identical_views(a, b):
            ret.append(a)
        else:
            ret.append(b)
    return ret
