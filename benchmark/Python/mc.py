from __future__ import print_function
import util
B = util.Benchmark()
if B.bohrium:
    import bohrium as np
else:
    import numpy as np

def montecarlo_pi(N, I):
    acc=0.0
    for i in xrange(I):
        x = np.array(np.random.random(N), dtype=B.dtype)
        y = np.array(np.random.random(N), dtype=B.dtype)

        z = np.sqrt(x**2 + y**2) <= 1.0
        acc += np.sum(z) * 4.0 / N

    acc /= I
    return acc

def main():
    N, I = B.size
    B.start()
    R = montecarlo_pi(N, I)
    B.stop()
    B.pprint()

if __name__ == "__main__":
    main()
