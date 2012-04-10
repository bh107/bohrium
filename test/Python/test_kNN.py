import numpy as np
import numpytest

def compute_targets(base, targets):
    b1 = base[:,np.newaxis,:]
    d1 = (b1-targets)**2
    d2 = np.add.reduce(d1, 2)
    d3 = np.sqrt(d2)
    r  = np.max(d2, axis=1)
    return r

def kNN(src, dist):
    targets = np.array(src, dtype=float, dist=dist)
    base    = np.array(src, dtype=float, dist=dist)
    return compute_targets(base, targets)

def run():
    db_length = 100
    ndims = 64
    src = numpytest.random_list((db_length, ndims))
    Seq = kNN(src, False)
    Par = kNN(src, True)

    if not numpytest.array_equal(Seq,Par):
        raise Exception("Uncorrect result matrix\n")

if __name__ == "__main__":
    run()
