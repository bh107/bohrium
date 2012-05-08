import numpy as np

def dot(A,B):
    Adim = A.ndim
    Bdim = B.ndim
    if Adim == 1:
        A = A[np.newaxis,:]
    if Bdim == 1:
        B = B[:,np.newaxis]
    res = np.add.reduce(A[:,np.newaxis]*np.transpose(B),2)
    if Bdim == 1:
        res = res[:,0]
    if Adim == 1:
        res = res[0]
    return res

def solve(A ,b , cphvb=True):
    # solve Ax=b via Gausian elimination
    W = np.hstack((A,b[:,np.newaxis]))
    if cphvb:
        cnp.handle_array(W)    
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
