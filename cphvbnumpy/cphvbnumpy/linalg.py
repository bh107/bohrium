"""
LinAlg
~~~~~~

Common linear algebra functions

"""
import cphvbnumpy as np
import numpy.linalg as la
from numpy.linalg import *

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

    >>> import cphvbnumpy as np
    >>> a = np.array([[3,1], [1,2]])
    >>> b = np.array([9,8])
    >>> x = np.linalg.solve(a, b)
    >>> x
    array([ 2.,  3.])

    Check that the solution is correct:

    >>> (np.dot(a, x) == b).all()
    True 
    """

    W = np.hstack((a,b[:,np.newaxis]))
    for p in xrange(W.shape[0]-1):
        for r in xrange(p+1,W.shape[0]):
            W[r] = W[r] - W[p]*(W[r,p]/W[p,p])
    x = np.empty(np.shape(b), dtype=b.dtype, cphvb=b.cphvb)
    c = b.size
    for r in xrange(c-1,0,-1):
        x[r] = W[r,c]/W[r,r]
        W[0:r,c] = W[0:r,c] - W[0:r,r] * x[r]
    x[0] = W[0,c]/W[0,0]
    return x

def jacobi(a, b, tol=0.0005):
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

    >>> import cphvbnumpy as np
    >>> a = np.array([[3,1], [1,2]])
    >>> b = np.array([9,8])
    >>> x = np.linalg.jacobi(a, b)
    >>> x
    array([ 2.,  3.])

    Check that the solution is correct:

    >>> (np.dot(a, x) == b).all()
    True 
    """
    x = np.ones(np.shape(b), dtype=b.dtype, cphvb=b.cphvb)
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


def lu(A):
    """
    Compute the LU decomposition.

    The decomposition is::

        A = P L U

    where P is a permutation matrix, L lower triangular with unit
    diagonal elements, and U upper triangular.

    Parameters
    ----------
    A : array, shape (M, M)
        Array to decompose

    Returns
    -------
    lu : array, shape (M, M)
         2d array containing L in the over triangular part, except the unit
         diagonal, and U in the upper triangular part
    
    p : array, shape (M)
        Contains the row pivots used by the decomposition. 
        Row i have been swaped with p[i]
    """

    if A.dtype != numpy.float32 and A.dtype != numpy.float64:
        raise ValueError("Input must be floating point numbers")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be square 2-d.")
    if A.cphvb:
        LU = A.copy() #do not overwrite original A
        P = empty((A.shape[0],), dtype=numpy.int32)
        bridge.lu(LU,P)
        return (LU, P)
    else:
        raise ValueError("LU is only supported for cphvb-enabled arrays")

