"""
LinAlg
~~~~~~

Common linear algebra functions

"""
import bohrium as np
import bohrium.blas as blas
import numpy_force.linalg as la
import numpy_force as numpy

# We import all of NumPy LinAlg and overwrite with the objects we implement ourself
from numpy_force.linalg import *

from . import bhary
from . import ufuncs
from . import array_create
from ._util import dtype_equal
from .bhary import fix_biclass_wrapper


@fix_biclass_wrapper
def gauss(a):
    """
    Performe Gausian elimination on matrix a without pivoting
    """
    for c in range(1, a.shape[0]):
        a[c:, c - 1:] = a[c:, c - 1:] - (a[c:, c - 1] / a[c - 1, c - 1:c])[:, None] * a[c - 1, c - 1:]
        np.flush()
    a /= np.diagonal(a)[:, None]
    return a


@fix_biclass_wrapper
def lu(a):
    """
    Performe LU decomposition on the matrix a so A = L*U
    """
    u = a.copy()
    l = np.zeros_like(a)
    np.diagonal(l)[:] = 1.0
    for c in range(1, u.shape[0]):
        l[c:, c - 1] = (u[c:, c - 1] / u[c - 1, c - 1:c])
        u[c:, c - 1:] = u[c:, c - 1:] - l[c:, c - 1][:, None] * u[c - 1, c - 1:]
        np.flush()
    return (l, u)


@fix_biclass_wrapper
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

    w = gauss(np.hstack((a, b[:, np.newaxis])))
    lc = w.shape[1] - 1
    x = w[:, lc].copy()
    for c in range(lc - 1, 0, -1):
        x[:c] -= w[:c, c] * x[c:c + 1]
        np.flush()
    return x


@fix_biclass_wrapper
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
    D = 1 / np.diag(a)
    R = np.diag(np.diag(a)) - a
    T = D[:, np.newaxis] * R
    C = D * b
    error = tol + 1
    while error > tol:
        xo = x
        x = np.add.reduce(T * x, -1) + C
        error = norm(x - xo) / norm(x)
    return x


@fix_biclass_wrapper
def matmul(a, b, no_blas=False):
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
    if not dtype_equal(a, b):
        raise ValueError("Input must be of same type")

    if a.ndim != 2 and b.ndim != 2:
        raise ValueError("Input must be 2-D.")

    if not (bhary.check(a) or bhary.check(b)):  # Both are regular numpy arrays
        return numpy.dot(a, b)
    else:
        a = array_create.array(a)
        b = array_create.array(b)

    # If the dtypes are both float, we can use BLAS to calculate
    # the dot-product, if BLAS is present.
    if not no_blas and a.dtype.kind in np.typecodes["AllFloat"] and b.dtype.kind in np.typecodes["AllFloat"]:
        try:
            return blas.gemm(a, b)
        except:
            pass

    return ufuncs.add.reduce(a[:, numpy.newaxis] * numpy.transpose(b), -1)


@fix_biclass_wrapper
def dot(a, b, no_blas=False):
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
    if not (bhary.check(a) or bhary.check(b)):  # Both are regular numpy arrays
        return numpy.dot(a, b)
    else:
        a = array_create.array(a)
        b = array_create.array(b)

    if b.ndim == 1:
        return ufuncs.add.reduce(a * b, -1)

    if a.ndim == 1:
        return ufuncs.add.reduce(a * numpy.transpose(b), -1)

    if not no_blas and a.ndim == 2 and b.ndim == 2:
        # If the dtypes are both float, we can use BLAS to calculate
        # the dot-product, if BLAS is present.
        if a.dtype.kind in np.typecodes["AllFloat"] and b.dtype.kind in np.typecodes["AllFloat"]:
            try:
                return blas.gemm(a, b)
            except:
                pass

    return ufuncs.add.reduce(a[:, numpy.newaxis] * numpy.transpose(b), -1)


@fix_biclass_wrapper
def norm(x, ord=None, axis=None):
    """
    This version of norm is not fully compliant with the NumPy version,
    it only supports computing 2-norm of a vector.
    """

    if ord != None:
        raise NotImplementedError("Unsupported value param ord=%s" % ord)
    if axis != None:
        raise NotImplementedError("Unsupported value of param ord=%s" % axis)

    r = np.sum(x * x)
    if issubclass(np.dtype(r.dtype).type, np.integer):
        r_f32 = np.empty(r.shape, dtype=np.float32)
        r_f32[:] = r
        r = r_f32
    return np.sqrt(r)


@fix_biclass_wrapper
def tensordot(a, b, axes=2):
    """
    Compute tensor dot product along specified axes for arrays >= 1-D.

    Given two tensors (arrays of dimension greater than or equal to one),
    `a` and `b`, and an array_like object containing two array_like
    objects, ``(a_axes, b_axes)``, sum the products of `a`'s and `b`'s
    elements (components) over the axes specified by ``a_axes`` and
    ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N``
    dimensions of `a` and the first ``N`` dimensions of `b` are summed
    over.

    Parameters
    ----------
    a, b : array_like, len(shape) >= 1
        Tensors to "dot".
    axes : variable type
        * integer_like scalar
          Number of axes to sum over (applies to both arrays); or
        * (2,) array_like, both elements array_like of the same length
          List of axes to be summed over, first sequence applying to `a`,
          second to `b`.

    See Also
    --------
    dot, einsum

    Notes
    -----
    When there is more than one axis to sum over - and they are not the last
    (first) axes of `a` (`b`) - the argument `axes` should consist of
    two sequences of the same length, with the first axis to sum over given
    first in both sequences, the second axis second, and so forth.

    Examples
    --------
    A "traditional" example:

    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> c = np.tensordot(a,b, axes=([1,0],[0,1]))
    >>> c.shape
    (5, 2)
    >>> c
    array([[ 4400.,  4730.],
           [ 4532.,  4874.],
           [ 4664.,  5018.],
           [ 4796.,  5162.],
           [ 4928.,  5306.]])
    >>> # A slower but equivalent way of computing the same...
    >>> d = np.zeros((5,2))
    >>> for i in range(5):
    ...   for j in range(2):
    ...     for k in range(3):
    ...       for n in range(4):
    ...         d[i,j] += a[k,n,i] * b[n,k,j]
    >>> c == d
    array([[ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True]], dtype=bool)

    An extended example taking advantage of the overloading of + and \\*:

    >>> a = np.array(range(1, 9))
    >>> a.shape = (2, 2, 2)
    >>> A = np.array(('a', 'b', 'c', 'd'), dtype=object)
    >>> A.shape = (2, 2)
    >>> a; A
    array([[[1, 2],
            [3, 4]],
           [[5, 6],
            [7, 8]]])
    array([[a, b],
           [c, d]], dtype=object)

    >>> np.tensordot(a, A) # third argument default is 2
    array([abbcccdddd, aaaaabbbbbbcccccccdddddddd], dtype=object)

    >>> np.tensordot(a, A, 1)
    array([[[acc, bdd],
            [aaacccc, bbbdddd]],
           [[aaaaacccccc, bbbbbdddddd],
            [aaaaaaacccccccc, bbbbbbbdddddddd]]], dtype=object)

    >>> np.tensordot(a, A, 0) # "Left for reader" (result too long to incl.)
    array([[[[[a, b],
              [c, d]],
              ...

    >>> np.tensordot(a, A, (0, 1))
    array([[[abbbbb, cddddd],
            [aabbbbbb, ccdddddd]],
           [[aaabbbbbbb, cccddddddd],
            [aaaabbbbbbbb, ccccdddddddd]]], dtype=object)

    >>> np.tensordot(a, A, (2, 1))
    array([[[abb, cdd],
            [aaabbbb, cccdddd]],
           [[aaaaabbbbbb, cccccdddddd],
            [aaaaaaabbbbbbbb, cccccccdddddddd]]], dtype=object)

    >>> np.tensordot(a, A, ((0, 1), (0, 1)))
    array([abbbcccccddddddd, aabbbbccccccdddddddd], dtype=object)

    >>> np.tensordot(a, A, ((2, 1), (1, 0)))
    array([acccbbdddd, aaaaacccccccbbbbbbdddddddd], dtype=object)

    """
    try:
        iter(axes)
    except:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

    a, b = np.array(a), np.array(b)
    as_ = a.shape
    nda = len(a.shape)
    bs = b.shape
    ndb = len(b.shape)
    equal = True
    if (na != nb):
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (-1, N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, -1)
    oldb = [bs[axis] for axis in notin]

    at = a.transpose(newaxes_a).reshape(newshape_a)
    bt = b.transpose(newaxes_b).reshape(newshape_b)
    res = dot(at, bt)
    return res.reshape(olda + oldb)


@fix_biclass_wrapper
def solve_tridiagonal(a, b, c, rhs):
    """
    Solver for tridiagonal systems,

    ..math::
        A x = b

    based on the Thomas algorithm (not unconditionally stable).

    If the input arrays have more than one dimension, solutions are computed along the last axis.
    Systems are solved in parallel if OpenMP is present. All inputs must have equal shape, and
    be of dtype `float32` or `float64`.

    :param a: Lower diagonal elements. a[...,0] is not used.
    :param b: Main diagonal elements.
    :param c: Upper diagonal elements. c[...,-1] is not used.
    :param rhs: Solution vector.
    :returns: Solution of the tridiagonal system(s). Has the same shape as the input arrays.
    """
    if not (a.shape == b.shape == c.shape == rhs.shape):
        raise ValueError("All inputs must have equal shapes")

    if a.shape[-1] < 2:
        raise ValueError("Last axis must contain at least 2 elements")

    out_shape = a.shape
    num_systems = 1
    if a.ndim > 1:
        for s in out_shape[:-1]:
            num_systems *= s
    system_size = out_shape[-1]

    diagonals = array_create.empty((3, num_systems, system_size), dtype=rhs.dtype, bohrium=True)
    diagonals[0] = a.reshape(num_systems, system_size)
    diagonals[1] = b.reshape(num_systems, system_size)
    diagonals[2] = c.reshape(num_systems, system_size)
    rhs = rhs.reshape(num_systems, system_size)
    out = array_create.zeros_like(rhs)
    ufuncs.extmethod("tdma", out, diagonals, rhs)
    return out.reshape(out_shape)


@fix_biclass_wrapper
def cg(A, b, x=None, tol=1e-5, force_niter=None):
    """
    Conjugate Gradient (CG) solver

    Implemented as example MATLAB code from <https://en.wikipedia.org/wiki/Conjugate_gradient_method>
    """
    # If no guess is given, set an empty guess
    if x is None:
        x = array_create.zeros_like(b)

    r = b - dot(A, x)
    p = r.copy()
    r_squared = r * r
    rsold = np.sum(r_squared)

    tol_squared = tol * tol
    i = 0
    while np.max(r_squared) > tol_squared or force_niter is not None:
        Ap = dot(A, p)
        alpha = rsold / dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        r_squared = r * r
        rsnew = np.sum(r_squared)

        p = r + (rsnew / rsold) * p
        rsold = rsnew
        if force_niter is not None and i >= force_niter:
            break
        i += 1
    return x
