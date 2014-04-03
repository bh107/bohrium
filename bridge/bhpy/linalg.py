"""
LinAlg
~~~~~~

Common linear algebra functions

"""
import bohrium as np
import numpy.linalg as la

def gauss(a, b):
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

    w = np.hstack((a,b[:,np.newaxis]))
    # Transform w to row echelon form
    for r in xrange(1,W.shape[0]):
        w[r] = w[r] - w[r-1]*(w[r,r-1]/w[r-1,r-1])
    x = np.empty_like(b)
    c = x.size
    for r in xrange(c-1,0,-1):
        x[r] = w[r,c]/w[r,r]
        w[0:r,c] = w[0:r,c] - w[0:r,r] * x[r]
    x[0] = w[0,c]/w[0,0]
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
