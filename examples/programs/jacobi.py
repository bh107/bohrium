import numpy as np
import cphvbnp as cnp
import linalg as cla
import 
import numpy.linalg as nla

def jacobi(A, b, cphvb ,tol=0.0005):
    # solve Ax=b via the Jacobi method
    x = cnp.ones(np.shape(b), dtype=b.dtype, cphvb=b.cphvb)
    D = cnp.diag(1/np.diag(A))
    R = cnp.diag(np.diag(A)) - A
    T = cla.dot(D,R)
    C = cla.dot(D,b)
    error = tol + 1
    while error > tol:
        xo = x
        x = cla.dot(T,x) + C
        error = nla.norm(x-xo)
    return x

