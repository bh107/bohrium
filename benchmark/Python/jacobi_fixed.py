from __future__ import print_function
import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np
from bohrium.stdviews import no_border, grid, diagonals

def jacobi_fixed_init(size):

    v    = np.arange(size)
    data = (v*(v[np.newaxis,:].T+2)+2)/size

    return data

def jacobi_fixed(data, iterations):

    active  = no_border(data, 1)
    g       = grid(data, 1)
    d       = diagonals(data, 1)
    fak     = 1./20

    for _ in xrange(iterations):
        active[:] = (4*sum(g) + sum(d))*fak

    return data

def main():
    B = util.Benchmark()
    (N, I) = B.size
    data = jacobi_fixed_init(N)

    B.start()
    R = jacobi_fixed(data, I)
    B.stop()

    B.pprint()
    if B.verbose:
        print(R)
    if B.outputfn:
        B.tofile(B.outputfn, {'res': R})

if __name__ == "__main__":
    main()
