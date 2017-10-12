"""
Conjugate Gradient (CG) solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from . import array_create
from . import linalg
from . import summations


# Implemented as example MATLAB code from <https://en.wikipedia.org/wiki/Conjugate_gradient_method>
def cg(A, b, x=None, tol=1e-5, force_niter=None):
    # If no guess is given, set an empty guess
    if x is None:
        x = array_create.zeros_like(b)

    r = b - linalg.dot(A, x)
    p = r.copy()
    r_squared = r * r
    rsold = summations.sum(r_squared)

    tol_squared = tol * tol
    i = 0
    while summations.max(r_squared) > tol_squared or force_niter is not None:
        Ap = linalg.dot(A, p)
        alpha = rsold / linalg.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        r_squared = r * r
        rsnew = summations.sum(r_squared)

        p = r + (rsnew / rsold) * p
        rsold = rsnew
        if force_niter is not None and i >= force_niter:
            break
        i += 1
    return x
