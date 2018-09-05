"""
Array manipulation routines
===========================
"""
from copy import deepcopy
from . import array_create
import numpy_force as numpy
from . import bhary
from . import _util
from .bhary import fix_biclass_wrapper
from . import numpy_backport
from . import loop


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
    return numpy_backport.as_strided(ary, shape=ret_shape, strides=ret_strides)


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
    size = d.size + abs(k)
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

        if len(args) == 1:
            if numpy.isscalar(args[0]):  # It is possible that `args[0]` is a scalar
                shape = (1,)
            else:
                shape = args[0].shape
            return (args, shape)

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

        # The broadcasted view inherits dynamic changes if there are any.
        # Used for broadcasting dynamic views within a do_while loop
        bcast_array = args[0]
        bcast_dvi = bcast_array.bhc_dynamic_view_info
        for a, b in zip(args, bargs):
            # If the broadcast view changes shape between iterations,
            # force the same change to the views being broadcasted.
            # Used in regard to iterators within do_while loops.
            a_dvi = a.bhc_dynamic_view_info

            # If array that is broadcasted has dynamic changes, the
            # broadcasted array must inherit these
            if a_dvi:
                b_dvi = deepcopy(a_dvi)
            else:
                b_dvi = loop.DynamicViewInfo({}, a.shape, a.strides)

            # If the array that is broadcasted from has changes in shape
            # must these changes also be inherited by the broadcasted array
            if bcast_dvi:
                # If the view contains a slide in a broadcasted dimension,
                # the slide must be inherited
                for dim in bcast_dvi.dims_with_changes():
                    # If the array, which is broadcasted from, does not have
                    # dynamic changes, there are no changes to add
                    if bcast_dvi.dim_shape_change(dim) == 0:
                        continue

                    # The array, which is broadcasted from, has dynamic changes
                    # while the broadcasted array does not. Add the changes to the
                    # broadcasted array
                    elif b_dvi.dim_shape_change(dim) == 0:
                        for (_, shape_change, step_delay, shape, stride) in bcast_dvi.changes_in_dim(dim):
                            # No reason to add a change of 0 in the dimension
                            if shape_change == 0:
                                continue
                            b_dvi.add_dynamic_change(dim, 0, shape_change, step_delay, shape, stride)

                    # Both array, which is broadcasted from, and the broadcasted array has
                    # dynamic changes. Make sure they are the same. If not the change cannot
                    # be guessed, which results in an error.
                    elif b_dvi.dim_shape_change(dim) != 0 and \
                            b_dvi.dim_shape_change(dim) != bcast_dvi.dim_shape_change(dim):
                        raise loop.IteratorIllegalBroadcast(
                            dim, a.shape, a_dvi.dim_shape_change(dim),
                            bcast_array.shape, bcast_dvi.dim_shape_change(dim))

            # Add the dynamic changes, if any
            if b_dvi.has_changes():
                b.bhc_dynamic_view_info = b_dvi

            # Append the broadcasted array
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
