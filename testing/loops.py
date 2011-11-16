#!/usr/bin/env python
from pprint import pprint as pp
import numpy as np
import time
import sys

CPHVB=True
size = 1024

def main():

    x = np.array([1]*size, dtype=np.float32, dist=CPHVB)
    y = np.array([1]*size, dtype=np.float32, dist=CPHVB)
    z = np.empty(size, dtype=np.float32, dist=CPHVB)

    np.add( x, y, z )
    print len(z), z

if __name__ == "__main__":
    main()
