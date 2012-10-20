import cphvbnumpy as np
import cphvbbridge as cb
import util

def data_range(B, N, F):

    # Data is a list of N-dimensional coordinates / features in euclidean space.
    data_flat   = np.array(range(0,N+1)*F, dtype=B.dtype, cphvb=B.cphvb)
    data        = data_flat.reshape(F, N+1)
    data.cphvb  = B.cphvb

    # Target is a single coordinate / feature in N-dimensional euclidian-space.
    x = np.array([[N/2]]*F, dtype=B.dtype, cphvb=B.cphvb)
    x.cphvb     = B.cphvb

    return data, x

def data_image(B, N, F):

    try:
        import Image
        img     = Image.open("knn.input.bmp")
        data    = np.array(img.getdata()).T.copy()
        x       = np.array([[0],[250],[0]], cphvb=B.cphvb)
        data.cphvb = B.cphvb
        return data, x
    except Exception as e:
        print "Failed using image data-set, reverting to range. Err=[%s]" % e

    return data_range(B, N, F)

def main():

    B = util.Benchmark()
    N   = B.size[0]                             # Size of the dataset
    F   = B.size[1]                             # Number of features in the dataset
    K   = B.size[2] if B.size[2] < N else N     # The K number of neighbors to find

    #data, x = data_image(B, N, F)
    data, x = data_range(B, N, F)               # Grab a data-set
    F = len(x)
    cb.flush()                                  # Flush & Measure
    B.start()

    sqd = np.sqrt(((data - x)**2).sum(axis=0))  # The naive kNN-implementation

    B.stop()                                    # Print the timing results
    B.pprint()

    idx = np.argsort(sqd)                       # Get the indexes
    nn  = data[:,idx[:K]]                       # Get the corresponding elements

if __name__ == "__main__":
    main()
