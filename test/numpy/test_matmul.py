import cphvbnumpy as np
from numpytest import numpytest

class test_matmul(numpytest):
    def __init__(self,*args,**kwargs):
        numpytest.__init__(self,*args,**kwargs)
        self.config['maxerror'] = 0.0
        self.config['dtypes'] = [np.int32,np.int64,np.float32,np.float64]
    def test_matmul(self,setup):
        maxdim = 2
        for m in range(1,maxdim+1):
            for n in range(1,maxdim+1):
                for k in range(1,maxdim+1):
                    A = self.array((k,m),dtype=setup['dtype'])
                    B = self.array((m,k),dtype=setup['dtype'])
                    yield np.matmul(A,B)
    def test_dot(self,setup):
        maxdim = 6
        for m in range(1,maxdim+1):
            for n in range(1,maxdim+1):
                for k in range(1,maxdim+1):
                    A = self.array((k,m),dtype=setup['dtype'])
                    B = self.array((m,k),dtype=setup['dtype'])
                    yield np.dot(A,B)
