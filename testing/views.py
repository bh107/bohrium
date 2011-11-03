#!/usr/bin/env python
from pprint import pprint as pp
import numpy as np
import time
import sys

CPHVB=True
size = 1024

def main():

    stuff = \
    [
        [
            [4]*1024,
            [2]*1024,
            [8]*1024,
        ],
        [
            [4]*1024,
            [2]*1024,
            [8]*1024,
        ],
    ]

    x = np.array(stuff, dtype=np.float32, dist=CPHVB)

    y = x[0]
    z = x[1]
    print np.add( y, z )

if __name__ == "__main__":
    main()
