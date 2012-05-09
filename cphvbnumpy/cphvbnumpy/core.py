import numpy as np

def empty(shape, dtype=float, cphvb=True):
    return np.empty(shape, dtype, cphvb)

def ones(shape, dtype=float, cphvb=True):
    A = empty(shape, dtype, cphvb)
    A[:] = 1
    return A

def zeros(shape, dtype=float, cphvb=True):
    A = empty(shape, dtype, cphvb)
    A[:] = 0
    return A

def flatten(A):
    return A.reshape(np.multiply.reduce(np.asarray(A.shape)))

def diagonal(A,k=0):
    if A.ndim !=2 :
        raise Exception("diagonal only supports 2 dimensions\n")
    if k < 0:
        d = A[-k:,0]
    elif k > 0:
        d = A[0,k:]
    else:
        d = A[0]
    d.strides=(A.strides[0]+A.strides[1])
    return d

def diagflat(d,k=0):
    d = np.asarray(d)
    d = flatten(d) 
    A = zeros((d.size,d.size), dtype=d.dtype, cphvb=d.cphvb)
    Ad = diagonal(A)
    Ad[:] = d 
    return A

def diag(A,k=0):
    if A.ndim == 1:
        return diagflat(A,k)
    elif A.ndim == 2:
        return diagonal(A,k)
    else:
        raise ValueError("Input must be 1- or 2-d.")

def dot(A,B):
    if B.ndim == 1:
        return np.add.reduce(A*B,-1)
    if A.ndim == 1:
        return np.add.reduce(A*np.transpose(B),-1)
    return np.add.reduce(A[:,np.newaxis]*np.transpose(B),-1)
