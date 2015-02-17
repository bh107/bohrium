#!/usr/bin/env python
import bohrium as np

#a = np.arange(0, n)
def stuff(n):
    a = np.ones(n)
    b = a[::2]
    c = a[::2]
    
    return ((b+c)*b)/c

print(stuff(20))
