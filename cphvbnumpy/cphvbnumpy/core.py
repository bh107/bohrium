import numpy as np
from numpy import *
import cphvbbridge as bridge

def empty(shape, dtype=float, cphvb=True):
    return np.empty(shape, dtype=dtype, cphvb=cphvb)

def ones(shape, dtype=float, cphvb=True):
    A = empty(shape, dtype=dtype, cphvb=cphvb)
    A[:] = 1
    return A

def zeros(shape, dtype=float, cphvb=True):
    A = empty(shape, dtype=dtype, cphvb=cphvb)
    A[:] = 0
    return A

def flatten(A):
    return A.reshape(np.multiply.reduce(np.asarray(A.shape)))

def diagonal(A,offset=0):
    if A.ndim !=2 :
        raise Exception("diagonal only supports 2 dimensions\n")
    if offset < 0:
        offset = -offset
        if (A.shape[0]-offset) > A.shape[1]:
            d = A[offset,:]
        else:
            d = A[offset:,0]
    else:
         if A.shape[1]-offset > A.shape[0]:
             d = A[:,offset]
         else:
             d = A[0,offset:]
    d.strides=(A.strides[0]+A.strides[1])
    return d

def diagflat(d,k=0):
    d = np.asarray(d)
    d = flatten(d) 
    size = d.size+abs(k)
    A = zeros((size,size), dtype=d.dtype, cphvb=d.cphvb)
    Ad = diagonal(A, offset=k)
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
    if A.cphvb or B.cphvb:
        bridge.handle_array(A)
        bridge.handle_array(B)
    if B.ndim == 1:
        return np.add.reduce(A*B,-1)
    if A.ndim == 1:
        return add.reduce(A*np.transpose(B),-1)
    return add.reduce(A[:,np.newaxis]*np.transpose(B),-1)

def matmul(A,B):
    if A.dtype != B.dtype:
        raise ValueError("Input must be of same type")
    if A.ndim != 2 and B.ndim != 2:
        raise ValueError("Input must be 2-d.")
    if A.cphvb or B.cphvb:
        A.cphvb=True
        B.cphvb=True
        C = empty((A.shape[0],B.shape[1]),dtype=A.dtype)
        bridge.matmul(A,B,C)
        return C
    else:
	return np.dot(A,B)
