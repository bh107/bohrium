"""
Array manipulation routines
~~~~

"""
import array_create
import numpy
import ndarray

def flatten(A):
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
    return A.reshape(numpy.multiply.reduce(numpy.asarray(A.shape)))

def diagonal(A,offset=0):
    """
    Return specified diagonals.

    If `a` is 2-D, returns the diagonal of `a` with the given offset,
    i.e., the collection of elements of the form ``a[i, i+offset]``.

    Parameters
    ----------
    a : array_like
        Array from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal.  Can be positive or
        negative.  Defaults to main diagonal (0).

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
    >>> a = np.arange(4).reshape(2,2)
    >>> a
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
    """
    if A.ndim !=2 :
        raise Exception("diagonal only supports 2 dimensions\n")
    if offset < 0:
        offset = -offset
        if (A.shape[0]-offset) > A.shape[1]:
            d = A[offset,:]
        else:
            d = A[offset:,0]
    else:
         if A.shape[1]-offset > A.shape[0]:
             d = A[:,offset]
         else:
             d = A[0,offset:]
    d.strides=(A.strides[0]+A.strides[1],)
    return d

def diagflat(d,k=0):
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
    A = array_create.zeros((size,size), dtype=d.dtype, bohrium=ndarray.check(d))
    Ad = diagonal(A, offset=k)
    Ad[:] = d
    return A

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
        return diagflat(v,k)
    elif v.ndim == 2:
        return diagonal(v,k)
    else:
        raise ValueError("Input must be 1- or 2-d.")

###############################################################################
################################ UNIT TEST ####################################
###############################################################################

import unittest

class Tests(unittest.TestCase):
    pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
    unittest.TextTestRunner(verbosity=2).run(suite)
