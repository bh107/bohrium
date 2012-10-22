import cphvbnumpy as numpy
import util

def main():
    B = util.Benchmark()
    N, = B.size

    x = numpy.arange(N**2, dtype=numpy.float32)
    x.shape = (N, N)
    x.cphvb = B.cphvb

    y = numpy.arange(N**2, dtype=numpy.float32)
    y.shape = (N, N)
    x.cphvb = B.cphvb

    z = numpy.empty((N,N), dtype=numpy.float32)
    z.cphvb = B.cphvb

    B.start()
    for i in xrange(N):
        for j in xrange(N):
            col = x[i,:]
            row = y[:,j]
            v   = col * row 
            r   = numpy.add.reduce(v)
            #z[i,j] = r     # This is unsupported.

    B.stop()
    B.pprint()

if __name__ == "__main__":
    main()
