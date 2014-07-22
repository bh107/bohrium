import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np

def main():

    B = util.Benchmark()
    N, = B.size

    # Create matrices
    x = np.arange(N**2, dtype=np.float32)
    x.shape = (N, N)

    y = np.arange(N**2, dtype=np.float32)
    y.shape = (N, N)

    B.start()

    # Do the matrix multiplication
    np.add.reduce(x[:,np.newaxis] * np.transpose(y), -1)

    B.stop()
    B.pprint()

if __name__ == "__main__":
    main()
