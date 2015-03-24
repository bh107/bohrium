#!/usr/bin/env python
import sys

def test_reduce(np, bohrium, shape):

    return np.sum(np.ones(shape))

if __name__ == "__main__":

    """
    print("NUMPY")
    import numpy as np
    print test_reduce(np, False, (10, 10, 10))
    print test_reduce(np, False, (10, 10, 10))
    """

    print("BOHRIUM")
    import bohrium as np
    shape = (10,10,10)
    a = np.sum(np.ones(shape))
    b = np.sum(np.ones(shape))
    c = np.add.reduce(np.add.reduce(np.add.reduce(np.ones(shape))))
    d = np.sum(np.ones(shape))
    print a,b,c,d

    #print np.add.reduce(np.add.reduce(np.add.reduce(np.ones(shape))))

    #t1 = np.add.reduce(np.ones(shape))
    #t2 = np.add.reduce(t1)
    #t3 = np.add.reduce(t2)
    #del t1, t2
    #print t3
