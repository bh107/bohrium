import cphvbnumpy as np
from numpytest import numpytest

class test_atlas(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.005
    def test_matmul(self):
        niter = 6
        for m in range(1,niter+1):
            for n in range(1,niter+1):
                for k in range(1,niter+1):
                    A = self.array((k,m))
                    B = self.array((m,k))
                    yield np.dot(A,B)
