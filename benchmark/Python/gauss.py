import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np
import bohrium.linalg as la

def main():
    B = util.Benchmark()
    N = B.size[0]

    a = np.array(np.random.random((N, N)), dtype=B.dtype)

    B.start()
    la.gauss(a)
    B.stop()
    B.pprint()

if __name__ == "__main__":
    main()
