#!/usr/bin/env python
import numpy as np
import cphvbnumpy as cnp
import time
import sys

CPHVB   = True
size    = 1024*1024*40

try:
    CPHVB   = int(sys.argv[1])
    size    = int(sys.argv[2])
except:
    pass

def main():

    x = np.array([1]*size, dtype=np.float64, dist=CPHVB)
    y = 1
    z = np.empty((size), dtype=np.float64, dist=CPHVB)

    np.add( x, y, z )

if __name__ == "__main__":
    main()
