#!/usr/bin/env python
import bohrium as np

#a = np.arange(0, n)
def test_range(n):
    a = np.ones(n)
    b = a
    c = a
    
    return ((b+c)*b)/c

def test_random(n):
    a = np.random(n)
    b = a[::2]
    c = a[::2]
    
    return ((b+c)*b)/c

def test_ones(n):
    a = np.ones(n)
    b = a[::2]
    c = a[::2]
    
    return ((b+c)*b)/c

if __name__ == "__main__":
    print(test_range(20))
