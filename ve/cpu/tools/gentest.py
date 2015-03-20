#!/usr/bin/env python
import sys

def test_gauss(np,bohrium, shape):
    a = np.arange(5, shape[0]*shape[1]+5,dtype=np.float32).reshape(shape)
    for c in xrange(1, a.shape[0]):
        a[c:,c-1:] = a[c:,c-1:] - (a[c:,c-1]/a[c-1,c-1:c])[:,None] * a[c-1,c-1:]
        if bohrium:
            np.flush(a)
    a /= np.diagonal(a)[:,None]

    return a

if __name__ == "__main__":

    print("NUMPY")
    import numpy as np
    print test_gauss(np, False, (4,4))

    print("BOHRIUM")
    import bohrium as np
    print test_gauss(np, True, (4,4))

