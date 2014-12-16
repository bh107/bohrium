import argparse
import pprint
import time
import os
import numpy as np

import bohrium as bh

def sample(args):

    a = np.ones(args.shape)
    b = np.ones(args.shape)

    for i in xrange(0, args.iterations):
        b = a + b 

    return b

def main():
    p = argparse.ArgumentParser('Run a dummy program')
    p.add_argument(
        'shape', metavar='N', type=int,
        nargs='+', help="Shape of the input."
    )
    p.add_argument(
        'iterations', metavar='I', type=int, help="Number of iterations to run."
    )
    args = p.parse_args()
    bh.flush()
    start = time.time()
    res = sample(args)
    bh.flush()
    print(time.time() -start)

if __name__ == "__main__":
    main()
