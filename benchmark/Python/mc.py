import util
B = util.Benchmark()
if B.bohrium:
    import bohrium as np
else:
    import numpy as np

def montecarlo_pi(N, I):
    acc=0.0
    for i in xrange(I):
        x = np.random.random(N, dtype=B.dtype)
        y = np.random.random(N, dtype=B.dtype)

        z = np.sqrt(x*x+y*y)<=1.0
        acc += np.sum(z)*4.0/N

    acc /= I
    return acc

if __name__ == "__main__":
    N, I = B.size
    R = montecarlo_pi(N, I)
    B.start()
    B.stop()
    B.pprint()
