#!/usr/bin/env python
#import numpy as np
import bohrium as np

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
    a = np.ones(n)
    b = a + 1
    
    return np.sum(b) 

if __name__ == "__main__":
    print(test_reduce(20))
    #print(test_range(20))
    #print(test_random(20))
