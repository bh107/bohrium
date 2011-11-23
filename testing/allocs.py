#!/usr/bin/env python
from pprint import pprint as pp
import numpy as np
import time
import sys

CPHVB=True
size = 1024

def main():

    x = np.empty(1024, dtype=np.float32, dist=CPHVB)
    np.core.multiarray.random(x)
    print "Random Range\n", x, x[0]

if __name__ == "__main__":
    main()
