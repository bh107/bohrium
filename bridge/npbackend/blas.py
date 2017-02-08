"""
Basic Linear Algebra Subprograms (BLAS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Utilize BLAS directly from Python
"""

import bohrium as np
from sys import stderr
from . import ufuncs

def __blas(name, a, b, alpha=1.0, c=None, beta=0.0):
    if not b is None:
        if not (a.ndim == 2 and b.ndim == 2):
            stderr.write("[ext] Matrices need to be two-dimensional.\n")
            return None

        if a.shape[1] != b.shape[0]:
            stderr.write("[ext] Wrong shape of matrices: first argument has shape {} and second has shape {}.\n".format(a.shape, b.shape))
            return None

        if not b.flags['C_CONTIGUOUS']:
            b = b.copy()
    else:
        b = np.empty(shape=(a.shape[0], a.shape[1]), dtype=a.dtype)

    if not a.flags['C_CONTIGUOUS']:
        a = a.copy()

    if c is None:
        c = np.empty(shape=(a.shape[0], b.shape[1]), dtype=a.dtype)
    elif not c.flags['C_CONTIGUOUS']:
        c = c.copy()

    if alpha != 1.0:
        a = a * alpha

    if beta != 0.0:
        c = c * beta

    ufuncs.extmethod(name, c, a, b) # modifies 'c'
    return c

# All of A, B, and C are used
def gemm(a, b, alpha=1.0, c=None, beta=0.0):
    """ C := alpha * A * B + beta * C """
    return __blas("blas_gemm", a, b, alpha, c, beta)
