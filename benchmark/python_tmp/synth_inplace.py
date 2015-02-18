from __future__ import print_function
import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np

import bohrium as bh

def model(N, dtype=np.float32):
    """Construct some synthetic data-set to work on."""
    
    return np.ones(N, dtype=dtype)

def computation(X, I):
    """Compute something..."""

    N = X                   # Pseudo-grid
    S = X
    E = X
    W = X
    C = X
    for i in xrange(0, I):  # Pseudo-relaxation
        X[:] = N - S + E - W + C

    return X

def main():
    B = util.Benchmark()
    (N, I) = B.size
    
    if B.inputfn:
        X = B.load_array()
    else:
        X = model(N, dtype=B.dtype)

    if B.dumpinput:
        B.dump_arrays("synth_inplace", {'input': X})

    B.start()
    R = computation(X, I)
    B.stop()
    B.pprint()
    if B.verbose:
        print(np.sum(R))

if __name__ == "__main__":
    main()
