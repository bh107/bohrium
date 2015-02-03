#!/usr/bin/env python
from __future__ import print_function
import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np

def johnson(n, i):
    a = np.ones(n)
    b = a[::10]

    for _ in xrange(0, i):
        b += b + b + b

    return 0

def main():
    B = util.Benchmark()
    (N, I) = B.size
    B.start()
    R = johnson(N, I)
    B.stop()
    B.pprint()

if __name__ == "__main__":
    main()   
