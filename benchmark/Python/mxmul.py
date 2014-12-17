from __future__ import print_function
import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np

def main():

    B = util.Benchmark()
    N, = B.size

    # Load or create matrices
    if B.inputfn:
        arrays = B.load_arrays()
        x = arrays['x']
        y = arrays['y']
    else:
        x = np.arange(N**2, dtype=B.dtype)
        x.shape = (N, N)

        y = np.arange(N**2, dtype=B.dtype)
        y.shape = (N, N)

    if B.dumpinput:
        B.dump_arrays("mxmul", {'x': x, 'y': y})

    # Do the matrix multiplication
    B.start()
    R = np.add.reduce(x[:,np.newaxis] * np.transpose(y), -1)
    B.stop()

    # Print / dump
    B.pprint()
    if B.outputfn:
        B.tofile(B.outputfn, {'res': R})

if __name__ == "__main__":
    main()
