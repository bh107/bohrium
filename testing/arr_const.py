#!/usr/bin/env python
from pprint import pprint as pp
import numpy as np
import time
import sys

CPHVB=True
size = 1024

def arr_const():

    x = np.array([1]*size, dtype=np.float32, dist=CPHVB)
    print 2 + x
    print x + 2
    print x + x

if __name__ == "__main__":
    main()
