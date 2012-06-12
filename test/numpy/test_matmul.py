import cphvbnumpy as np
from numpytest import numpytest

class test_matmul(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0
        self.config['dtypes'] = [np.float32,np.float64,np.int64]

    def test_matmul(self):
        maxdim = 2
        for m in range(1,maxdim+1):
            for n in range(1,maxdim+1):
                for k in range(1,maxdim+1):
                    A = self.array((m,k))
                    B = self.array((k,n))
                    cmd = "array(({0},{1})),array(({1},{2}))".format(m,k,n)
                    yield (np.matmul(A,B),cmd)

    def test_dot(self):
        maxdim = 6
        for m in range(1,maxdim+1):
            for n in range(1,maxdim+1):
                for k in range(1,maxdim+1):
                    A = self.array((m,k))
                    B = self.array((k,n))
                    cmd = "array(({0},{1})),array(({1},{2}))".format(m,k,n)
                    yield (np.dot(A,B),cmd)
