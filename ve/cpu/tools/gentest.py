#!/usr/bin/env python
import sys

if len(sys.argv) > 1:
    import bohrium as np
else:
    import numpy as np

def test_range(n):
    a = np.arange(1,n+1)
    b = a
    c = a
    r = ((b+c)*b)/c
    return r

def test_range(n):
    a = np.arange(1,n+1)
    b = a[::2]
    c = a[::2]
    r = ((b+c)*b)/c
    return r

def test_random(n):
    a = np.random.random(n)
    b = a
    c = a
    r = ((b+c)*b)/c
    return r

def test_ones(n):
    a = np.ones((n,n,n,n))
    #a = np.ones((n, n,n))
    b = a[::2]
    c = a[::2]
    
    return ((b+c)*b)/c

def test_reduce(n):
    a = np.asarray(
    [  8.74143481e-01,   5.95615208e-01,   6.19592845e-01,   4.99800116e-01,
       3.60308260e-01,   9.47238982e-01,   2.68870294e-01,   6.87112629e-01,
          5.45386188e-02,   2.58751588e-05], dtype=np.float32)

    return np.add.reduce(a, axis=-1) 
    #return np.add.reduce(b) 

if __name__ == "__main__":
    print(test_reduce(2000))
    #print(test_range(20))
    #print(test_random(20))
