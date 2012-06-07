import cphvbnumpy as np
from numpytest import numpytest
import cphvbnumpy.linalg as la
from cphvbnumpy.examples import gameoflife, jacobi_stencil, k_nearest_neighbor as knn

class test_jacobi(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.config['dtypes'] = [np.float32,np.float64]
        self.size = 20

    def test_jacobi(self):
        A = self.array((self.size,self.size))
        B = self.array((self.size))
        A += np.diag(np.add.reduce(A,-1)) #make sure A is diagonally dominant
        return [la.jacobi(A,B)]

    def test_jacobi_stencil(self):
        grid = jacobi_stencil.frezetrap(self.size,self.size,cphvb=self.runtime['cphvb'])
        res = jacobi_stencil.solve(grid)
        return [res]
        
class test_gameoflife(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['dtypes'] = [np.int32]
        self.input = gameoflife.randomstate(10,10,dtype=np.int32,cphvb=False)

    def test_gameoflife(self):
        self.input.cphvb = self.runtime['cphvb']
        res   = gameoflife.play(self.input, 50)
        return [res]

