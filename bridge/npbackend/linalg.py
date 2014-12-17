"""
LinAlg
~~~~~~

Common linear algebra functions

"""
import bohrium as np
import numpy_force.linalg as la
import numpy_force as numpy
from . import ndarray
from . import ufunc
from . import array_create
from ._util import dtype_equal

def gauss(a):
    """
    Performe Gausian elimination on matrix a without pivoting
    """
    for c in xrange(1,a.shape[0]):
        a[c:,c-1:] = a[c:,c-1:] - (a[c:,c-1]/a[c-1,c-1:c])[:,None] * a[c-1,c-1:]
        np.flush(a)
    a /= np.diagonal(a)[:,None]
    return a


def lu(a):
    """
    Performe LU decomposition on the matrix a so A = L*U
    """
    u = a.copy()
    l = np.zeros_like(a)
    np.diagonal(l)[:] = 1.0
    for c in xrange(1,u.shape[0]):
        l[c:,c-1] = (u[c:,c-1]/u[c-1,c-1:c])
        u[c:,c-1:] = u[c:,c-1:] - l[c:,c-1][:,None] * u[c-1,c-1:]
        np.flush(u)
    return (l,u)


def solve(a, b):
    """
    Solve a linear matrix equation, or system of linear scalar equations
    using Gausian elimination.

    :param a: Coefficient matrix
    :type a:  array_like, shape (M, M)
    :param b: Ordinate or "dependent variable" values
    :type b:  array_like, shape (M,) or (M, N)

    :return:  Solution to the system a x = b
    :rtype:   ndarray, shape (M,) or (M, N) depending on b

    :raises: :py:exc:`LinAlgError` If `a` is singular or not square.

    **Examples:**
    Solve the system of equations ``3 * x0 + x1 = 9`` and ``x0 + 2 * x1 = 8``:

    >>> import bohrium as np
    >>> a = np.array([[3.,1.], [1.,2.]])
    >>> b = np.array([9.,8.])
    >>> x = np.linalg.solve(a, b)
    >>> x
    array([ 2.,  3.])

    Check that the solution is correct:

    >>> (np.dot(a, x) == b).all()
    True
    """
    if not (len(a.shape) == 2 and a.shape[0] == a.shape[1]):
        raise la.LinAlgError("a is not square")

    w = gauss(np.hstack((a,b[:,np.newaxis])))
    lc = w.shape[1]-1
    x = w[:,lc].copy()
    for c in xrange(lc-1,0,-1):
        x[:c] -= w[:c,c] * x[c:c+1]
        np.flush(x)
    return x

def jacobi(a, b, tol=0.0005):
    raise NotImplementedError("norm() isn't implemented")
    """
    Solve a linear matrix equation, or system of linear scalar equations
    using the Jacobi Method.

    :param a: Coefficient matrix
    :type a:  array_like, shape (M, M)
    :param b: Ordinate or "dependent variable" values
    :type b:  array_like, shape (M,) or (M, N)

    :return:  Solution to the system a x = b
    :rtype:   ndarray, shape (M,) or (M, N) depending on b

    :raises: :py:exc:`LinAlgError` If `a` is singular or not square.

    **Examples:**
    Solve the system of equations ``3 * x0 + x1 = 9`` and ``x0 + 2 * x1 = 8``:

    >>> import bohrium as np
    >>> a = np.array([[3,1], [1,2]])
    >>> b = np.array([9,8])
    >>> x = np.linalg.jacobi(a, b)
    >>> x
    array([ 2.,  3.])

    Check that the solution is correct:

    >>> (np.dot(a, x) == b).all()
    True
    """
    x = np.ones_like(b)
    D = 1/np.diag(a)
    R = np.diag(np.diag(a)) - a
    T = D[:,np.newaxis]*R
    C = D*b
    error = tol + 1
    while error > tol:
        xo = x
        x = np.add.reduce(T*x,-1) + C
        error = norm(x-xo)/norm(x)
    return x

def matmul(a,b):
    """
    Matrix multiplication of two 2-D arrays.

    Parameters
    ----------
    a : array_like
        First argument.
    b : array_like
        Second argument.

    Returns
    -------
    output : ndarray
        Returns the matrix multiplication of `a` and `b`.

    Raises
    ------
    ValueError
        If the last dimension of `a` is not the same size as
        the second-to-last dimension of `b`.

    See Also
    --------
    dot : Dot product of two arrays.

    Examples
    --------
    >>> np.matmul(np.array([[1,2],[3,4]]),np.array([[5,6],[7,8]]))
    array([[19, 22],
           [43, 50]])
    """
    if not dtype_equal(a,b):
        raise ValueError("Input must be of same type")
    if a.ndim != 2 and b.ndim != 2:
        raise ValueError("Input must be 2-D.")
    if ndarray.check(a) or ndarray.check(b):
        a = array_create.array(a)
        b = array_create.array(b)
        c = np.empty((a.shape[0],b.shape[1]),dtype=a.dtype)
        ufunc.extmethod("matmul",c,a,b)
        return c
    else:
    	return numpy.dot(a,b)

def dot(a,b, no_matmul=False):
    """
    Dot product of two arrays.

    For 2-D arrays it is equivalent to matrix multiplication, and for 1-D
    arrays to inner product of vectors (without complex conjugation). For
    N dimensions it is a sum product over the last axis of `a` and
    the second-to-last of `b`::

        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    Parameters
    ----------
    a : array_like
        First argument.
    b : array_like
        Second argument.

    Returns
    -------
    output : ndarray
        Returns the dot product of `a` and `b`.  If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        an array is returned.

    Raises
    ------
    ValueError
        If the last dimension of `a` is not the same size as
        the second-to-last dimension of `b`.

    See Also
    --------
    vdot : Complex-conjugating dot product.
    tensordot : Sum products over arbitrary axes.
    einsum : Einstein summation convention.

    Examples
    --------
    >>> np.dot(3, 4)
    12

    Neither argument is complex-conjugated:

    >>> np.dot([2j, 3j], [2j, 3j])
    (-13+0j)

    For 2-D arrays it's the matrix product:

    >>> a = [[1, 0], [0, 1]]
    >>> b = [[4, 1], [2, 2]]
    >>> np.dot(a, b)
    array([[4, 1],
           [2, 2]])

    >>> a = np.arange(3*4*5*6).reshape((3,4,5,6))
    >>> b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
    >>> np.dot(a, b)[2,3,2,1,2,2]
    499128
    >>> sum(a[2,3,2,:] * b[1,2,:,2])
    499128

    """
    if ndarray.check(a) or ndarray.check(b):
        a = array_create.array(a)
        b = array_create.array(b)
    if b.ndim == 1:
        return ufunc.add.reduce(a*b,-1)
    if a.ndim == 1:
        return ufunc.add.reduce(a*numpy.transpose(b),-1)
    if (not no_matmul) and a.ndim == 2 and b.ndim == 2:
        return matmul(a,b)
    return ufunc.add.reduce(a[:,numpy.newaxis]*numpy.transpose(b),-1)
