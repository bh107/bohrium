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

def empty_like(a, dtype=None, cphvb=None):
    if dtype == None:
        dtype = a.dtype
    if cphvb == None:
        cphvb = a.cphvb
    return empty(a.shape, dtype, cphvb)

def zeros_like(a, dtype=None, cphvb=None):
    b = empty_like(a, dtype=dtype, cphvb=cphvb)
    b[:] = 0
    return b

def ones_like(a, dtype=None, cphvb=None):
    b = empty_like(a, dtype=dtype, cphvb=cphvb)
    b[:] = 1
    return b

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
	
def lu(A):
    if A.dtype != np.float32 and A.dtype != np.float64:
        raise ValueError("Input must be floating point numbers")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be square 2-d.")
    if A.cphvb:
        LU = A.copy() #do not overwrite original A
        P = empty((A.shape[0],), dtype=np.int32)
        bridge.lu(LU,P)
        return (LU, P)
    else:
	    raise ValueError("LU not supported for non cphvb numpy")
	    
def fft(A):
    if A.cphvb and A.ndim <= 2:
      if A.dtype == np.complex64 or A.dtype == np.complex128: #maybe do type conversions for others
        B = empty(A.shape,dtype=A.dtype)
        bridge.fft(A,B)
        return B
    
	return np.fft.fft(A)
	
def fft2(A):
    if A.cphvb and A.ndim == 2:
      if A.dtype == np.complex64 or A.dtype == np.complex128: #maybe do type conversions for others
        B = empty(A.shape,dtype=A.dtype)
        bridge.fft2(A,B)
        return B
    
	return np.fft.fft2(A)

def rad2deg(x, out=None):
    if out == None:
        out = 180 * x / pi
    else:
        out[:] = 180 * x / pi
    return out

def deg2rad(x, out=None):
    if out == None:
        out = x * pi / 180
    else:
        out[:] = x * pi / 180
    return out
        
def logaddexp(x1, x2, out=None):
    if out == None:
        out = log(exp(x1) + exp(x2))
    else:
        out[:] = log(exp(x1) + exp(x2))
    return out
    
def logaddexp2(x1, x2, out=None):
    if out == None:
        out = log2(exp2(x1) + exp2(x2))
    else:
        out[:] = log2(exp2(x1) + exp2(x2))
    return out
    
def modf(x, out1=None, out2=None):
    if out1 == None:
        out1 = mod(x,1.0)
    else:
        out1[:] = mod(x,1.0)
    if out2 == None:
        out2 = floor(x)
    else: 
        out2[:] = floor(x)
    return (out1, out2)
