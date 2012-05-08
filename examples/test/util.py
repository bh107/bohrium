import numpy as np
import cphvbnumpy as cnp
import random

def array_equal(A,B,maxerror=0.0):
    if type(A) is not type(B):
        return False
    elif (not type(A) == type(np.array([]))) and (not type(A) == type([])):
        if A == B:
            return True
        else:
            return False
    cnp.unhandle_array(A)
    cnp.unhandle_array(B)
    A = A.flatten()
    B = B.flatten()
    if not len(A) == len(B):
        return False
    for i in range(len(A)):
        delta = abs(A[i] - B[i])
        if delta > maxerror:
            print "Delta error:",delta
            return False
    return True

def random_list(dims):
    (val,unique) = _random_list(dims)
    return val

def _random_list(dims, unique=1):
    if len(dims) == 0:
        return (unique, unique + 1)
    list = []
    for i in range(dims[-1]):
        (val,unique) = _random_list(dims[0:-1], unique + i)
        list.append(val)
    return (list, unique + dims[-1])
