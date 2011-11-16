#!/usr/bin/env python
from pprint import pprint as pp
import numpy as np
import time
import sys

def test_arr_const( accel, size ):

    x = np.array([1]*size, dtype=np.float32, dist=accel)
    print 2 + x
    print x + 2
    print x + x

def main( args ):

    accel   = 1             # 1 = True, 0 = False
    size    = 4             # Usually vertice-dimensions, 4x4
    test    = 'arr_const'   # Function to run

    try:
        CPHVB   = int(sys.argv[1])
        size    = int(sys.argv[2])
        test    = sys.argv[3]
    except:
        pass

    test_arr_const( accel, size )

if __name__ == "__main__":
    main(sys.argv)
