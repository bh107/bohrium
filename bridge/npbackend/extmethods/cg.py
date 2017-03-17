"""
Conjugate Gradient (CG) solver
"""

import bohrium as np
import bohrium.blas as blas
import bohrium.linalg as LA

from sys import stderr
from . import ufuncs

# Implemented as example MATLAB code from https://en.wikipedia.org/wiki/Conjugate_gradient_method
def cg(A, b, x=None):
    # If no guess is given, set an empty guess
    if x is None:
        x = np.zeros(shape=b.shape, dtype=b.dtype)

    # Initialize arrays
    alpha = np.zeros(shape=(1,1), dtype=b.dtype)
    rsold = np.zeros(shape=(1,1), dtype=b.dtype)
    rsnew = np.zeros(shape=(1,1), dtype=b.dtype)

    r = b - np.matmul(A, x)
    p = r;
    rsold = blas.gemmt(r, r, c=rsold)

    for _ in range(b.shape[0]):
        Ap = np.matmul(A, p)
        alpha = rsold / blas.gemmt(p, Ap, c=alpha)

        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = blas.gemmt(r, r, c=rsnew)

        if LA.norm(rsnew) < 1e-10:
            return x

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x
