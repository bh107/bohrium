import cphvbnumpy as np
import numpy.linalg as la
from numpy.linalg import *

def solve(A ,b):
    # solve Ax=b via Gausian elimination
    W = np.hstack((A,b[:,np.newaxis]))
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

def jacobi(A, b, tol=0.0005):
    # solve Ax=b via the Jacobi method
    x = np.ones(np.shape(b), dtype=b.dtype, cphvb=b.cphvb)
    D = 1/np.diag(A)
    R = np.diag(np.diag(A)) - A
    T = D[:,np.newaxis]*R
    C = D*b
    error = tol + 1
    while error > tol:
        xo = x
        x = np.add.reduce(T*x,-1) + C
        error = norm(x-xo)/norm(x)
    return x
