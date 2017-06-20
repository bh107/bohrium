"""
Conjugate Gradient (CG) solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import bohrium as np

from sys import stderr
from . import ufuncs


# Implemented as example MATLAB code from https://en.wikipedia.org/wiki/Conjugate_gradient_method
def cg(A, b, x=None, tol=1e-5):
    # If no guess is given, set an empty guess
    if x is None:
        x = np.zeros(shape=b.shape, dtype=b.dtype)

    # Initialize arrays
    alpha = np.zeros(shape=(1, 1), dtype=b.dtype)
    rsold = np.zeros(shape=(1, 1), dtype=b.dtype)
    rsnew = np.zeros(shape=(1, 1), dtype=b.dtype)

    r = b - np.dot(A, x)
    p = r.copy()
    r_squared = r * r
    rsold = np.sum(r_squared)

    tol_squared = tol * tol
    while np.max(r_squared) > tol_squared:
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(p, Ap)

        x = x + alpha * p
        r = r - alpha * Ap
        r_squared = r * r
        rsnew = np.sum(r_squared)

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x
