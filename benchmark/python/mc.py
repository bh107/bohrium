from __future__ import print_function
import util
import bohrium as bh
B = util.Benchmark()
if B.bohrium:
    import bohrium as np
else:
    import numpy as np

def solve(N):
    x = bh.random.random((N,), dtype=B.dtype, bohrium=B.bohrium)
    y = bh.random.random((N,), dtype=B.dtype, bohrium=B.bohrium)
    z = np.sqrt(x**2 + y**2) <= 1.0
    return np.sum(z) * 4.0 / N

def montecarlo_pi(N, I):
    acc=0.0
    for i in xrange(I):
        acc += solve(N)
    acc /= I
    return acc

def main():
    N, I = B.size
    B.start()
    R = montecarlo_pi(N, I)
    B.stop()
    B.pprint()
    if B.verbose:
        print(R)

if __name__ == "__main__":
    main()
