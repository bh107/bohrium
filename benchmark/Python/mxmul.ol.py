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

    B.start()

    numpy.add.reduce(x[:,numpy.newaxis]*numpy.transpose(y),-1)

    B.stop()
    B.pprint()

if __name__ == "__main__":
    main()
