import cphvbnumpy as np
from numpytest import numpytest

class test_matmul(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0
        self.config['dtypes'] = [np.float32,np.float64]

    def test_matmul(self):
        maxdim = 2
        for m in range(1,maxdim+1):
            for n in range(1,maxdim+1):
                for k in range(1,maxdim+1):
                    A = self.array((k,m))
                    B = self.array((m,k))
                    yield np.matmul(A,B)

    def test_dot(self):
        maxdim = 6
        for m in range(1,maxdim+1):
            for n in range(1,maxdim+1):
                for k in range(1,maxdim+1):
                    A = self.array((k,m))
                    B = self.array((m,k))
                    yield np.dot(A,B)
