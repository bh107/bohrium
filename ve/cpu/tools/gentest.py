#!/usr/bin/env python
#import numpy as np
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
    #a = np.array([ 0.48126301,  0.8642754 ,  0.71703511,  0.96928668,  0.66822982, 0.17131269,  0.66362536,  0.03300979,  0.49494377,  0.91533864])
    #a = np.array([ 0.48126301])
    a = np.array([ 0.921345  ,  0.43139523,  0.21352844,  0.73983538,  0.0227413 ,
     0.4394924 ,  0.50938076,  0.97166544,  0.67383581,  0.04343996], dtype=np.float32)

    return np.sum(a, 0) 
    #return np.add.reduce(b) 

if __name__ == "__main__":
    print(test_reduce(2000))
    #print(test_range(20))
    #print(test_random(20))
