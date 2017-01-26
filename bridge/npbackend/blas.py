"""
Basic Linear Algebra Subprograms (BLAS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Utilize BLAS directly from Python
"""

import bohrium as np
from sys import stderr
from . import ufuncs, bhary, array_create

def gemm(a, b, c=None, alpha=1.0, beta=0.0):
    if a.shape[1] != b.shape[0]:
        stderr.write("[ext] Wrong shape of matrices: first argument has shape {} and second has shape {}.\n".format(a.shape, b.shape))
        return None

    if c is None:
        c = np.empty(shape=(a.shape[0], b.shape[1]), dtype=a.dtype)

    if alpha != 1.0:
        a = a * alpha

    if beta != 0.0:
        c = c * beta

    ufuncs.extmethod("blas_gemm", c, a, b)
    return c
