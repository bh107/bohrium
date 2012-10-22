import cphvbnumpy as np
import util

if __name__ == "__main__":

    B = util.Benchmark()
    I = B.size.pop()
    N = B.size.pop()

    # scattering n points over the unit square
    p = np.random.random((2,N))
    p.cphvb = B.cphvb

    B.start()

    # counting the points inside the unit circle
    for _ in xrange(0, I):
        idx = np.sqrt(p[0,:]**2+p[1,:]**2) < 1.0
        est = np.sum(idx)/float(N)*4.0

    B.stop()
    B.pprint()

