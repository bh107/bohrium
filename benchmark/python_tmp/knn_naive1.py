from __future__ import print_function
import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np

def data_range(B, N, F):
    """Pseudo-data setup using Python-range."""

    # Data is a list of N-dimensional coordinates / features in euclidean space.
    data_flat   = np.array(range(0,N+1)*F, dtype=B.dtype)
    data        = np.array(data_flat.reshape(F, N+1))

    # Target is a single coordinate / feature in N-dimensional euclidian-space.
    x = np.array([[N/2]]*F, dtype=B.dtype)

    return data, x

def data_image(B, N, F):
    """Data setup using a bitmap image."""

    try:
        import Image
        img     = Image.open("knn.input.bmp")
        data    = np.array(np.array(img.getdata()).T.copy())
        x       = np.array([[0],[250],[0]])
        return data, x
    except Exception as e:
        print("Failed using image data-set, reverting to range. Err=[%s]"%e)

    return data_range(B, N, F)

def data_random(B, N, F):
    """Pseudo-data setup using random."""

    # Data is a list of N-dimensional coordinates / features in euclidean space.
    data = np.random.random((F, N+1))

    # Target is a single coordinate / feature in N-dimensional euclidian-space.
    x = np.empty((F, 1))
    x[:,:] = 0.5

    return data, x

def main():

    B = util.Benchmark()
    N   = B.size[0]                             # Size of the dataset
    F   = B.size[1]                             # Number of features in the dataset
    K   = B.size[2] if B.size[2] < N else N     # The K number of neighbors to find

    #data, x = data_range(B, N, F)               # Grab a data-set
    data, x = data_random(B, N, F)

    B.start()

    sqd = np.sum(np.sqrt(((data - x)**2)))      # The naive kNN-implementation

    B.stop()                                    # Print the timing results

    B.pprint()
    if B.verbose:
        print(sqd)
    if B.outputfn:
        B.tofile(B.outputfn, {'res': sqd})

    #idx = np.argsort(sqd)                       # Get the indexes
    #nn  = data[:,idx[:K]]                       # Get the corresponding elements

if __name__ == "__main__":
    main()
