#!/usr/bin/env python
from pprint import pprint as pp
import numpy as np
import time
import sys

CPHVB=True
size = 1024

def main():

    x = np.array([1]*size, dtype=np.float32, dist=CPHVB)

    odd     = x[1::2]
    even    = x[0::2]
    print len(odd), len(even), np.add( odd, even )

if __name__ == "__main__":
    main()
