import util
import bohrium as np
from bohrium.stdviews import no_boarder, D3P27

def point27_init(fname='heat.npy', use_bohrium=True):
    return np.load(fname, bohrium=use_bohrium)

def point27(data, iterations):
    """TODO: Describe the benchmark."""
    
    active  = no_boarder(data, 1)
    stencil = D3P27(data)
    for _ in xrange(iterations):
        active[:] = sum(stencil)/27.0

    return active

if __name__ == "__main__":
    B = util.Benchmark()
    (I,) = B.size
    data = point27_init(use_bohrium=B.bohrium)
    B.start()
    result = point27(data, I, B.bohrium)
    B.stop()
    B.pprint()