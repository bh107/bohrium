import util
if util.Benchmark().bohrium:
    import bohrium as np
    import bohrium.linalg as la
else:
    import numpy as np
    import numpy.linalg as la

if __name__ == "__main__":

    B = util.Benchmark()
    N = B.size[0]

    a = np.random.random((N, N), dtype=B.dtype)

    B.start()
    la.gauss(a)
    B.stop()
    B.pprint()
