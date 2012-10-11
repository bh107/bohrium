import cphvbnumpy as np
import cphvbbridge as cb
import util

def main():

    B = util.Benchmark()
    N   = B.size[0]                             # Size of the dataset
    F   = B.size[1]                             # Number of features in the dataset
    K   = B.size[2] if B.size[2] < N else N     # The K number of neighbors to find

    # Data is a list of N-dimensional coordinates / features in euclidean space.
    data_flat   = np.array(range(0,N+1)*F, dtype=B.dtype)
    data        = data_flat.reshape(F, N+1)
    data.cphvb  = B.cphvb

    # Target is a single coordinate / feature in N-dimensional euclidian-space.
    x = np.array([[N/2]]*F, dtype=B.dtype)
    x.cphvb = B.cphvb

    cb.flush()                                  # Measuring...
    B.start()

    sqd = np.sqrt(((data - x)**2).sum(axis=0))  # The naive kNN-implementation
    idx = np.argsort(sqd)                       # Get the indexes
    nn  = data[:,idx[:K]]                       # Get the corresponding elements

    B.stop()                                    # Print the timing results
    B.pprint()

    print nn

if __name__ == "__main__":
    main()
