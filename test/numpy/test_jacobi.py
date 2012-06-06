import cphvbnumpy as np
from numpytest import numpytest
import cphvbnumpy.linalg as la
from cphvbnumpy.examples import jacobi_stencil

class test_matmul(numpytest):
    def __init__(self,*args,**kwargs):
        numpytest.__init__(self,*args,**kwargs)
        self.config['maxerror'] = 0.00001
        self.config['dtypes'] = [np.float32,np.float64]
        self.size = 20

    def test_jacobi(self,setup):
        A = self.array((self.size,self.size),dtype=setup['dtype'])
        B = self.array((self.size),dtype=setup['dtype'])
        A += np.diag(np.add.reduce(A,-1)) #make sure A is diagonally dominant
        return [la.jacobi(A,B)]

    def test_jacobi_stencil(self,setup):
        grid = jacobi_stencil.frezetrap(self.size,self.size,cphvb=setup['cphvb'])
        res = jacobi_stencil.solve(grid)
        return [res]
        
