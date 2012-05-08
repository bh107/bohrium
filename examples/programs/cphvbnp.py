import numpy as np

def ones(shape, dtype=np.float32, cphvb=True):
    A = np.empty(shape, dtype, cphvb)
    a[:] = 1

def zeros(shape, dtype=np.float32, cphvb=True):
    A = np.empty(shape, dtype, cphvb)
    a[:] = 0

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
        d = a[0]
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
