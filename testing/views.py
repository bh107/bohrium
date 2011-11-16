#!/usr/bin/env python
from pprint import pprint as pp
import numpy as np
import time
import sys

CPHVB=True
size = 1024

def main():

    x = np.array([x % 2 for x in range(0,size)], dtype=np.float32, dist=CPHVB)
    r = np.empty( size / 2, dtype=np.float32, dist=CPHVB )

    odd     = x[0::2]
    even    = x[1::2]

    print x, odd, even
    print np.add( odd, even )

if __name__ == "__main__":
    main()
