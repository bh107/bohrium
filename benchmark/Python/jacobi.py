import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np
from bohrium.stdviews import no_border, grid, diagonals

def jacobi_init(size):

    v    = np.arange(size)
    data = (v*(v[np.newaxis,:].T+2)+2)/size

    return data

def jacobi(data):

    active      = no_border(data,1)
    g           = grid(data,1)
    d           = diagonals(data,1)
    fak         = 1./20
    residual    = 110000000

    while residual>(10**-2) * (active.shape[0]**2):
        update    = (4*sum(g) + sum(d))*fak
        residual  = np.add.reduce(np.add.reduce(abs(update-active)))
        active[:] = update

    return data

if __name__ == "__main__":
    """
    """
    B = util.Benchmark()
    N, = B.size
    data = jacobi_init(N)
    data + 1   # Ensure that data is in the correct space.
    B.start()
    result = jacobi(data)
    B.stop()
    B.pprint()
