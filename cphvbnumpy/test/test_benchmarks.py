import cphvbnumpy as np
import cphvbnumpy.linalg as la
from numpy.testing import assert_array_equal, assert_almost_equal, run_module_suite
from numpy.random import randint, random
from cphvbnumpy.examples import gameoflife, jacobi_stencil, k_nearest_neighbor as knn

def test_dot():
    dtype = np.float32
    size = 50
    A = np.random.random((size,size), dtype=dtype, cphvb=False)
    B = np.random.random((size,size), dtype=dtype, cphvb=False)
    Ref = np.dot(A,B)
    A.cphvb=True
    B.cphvb=True
    cphVB = np.dot(A,B)
    cphVB.cphvb = False
    assert_array_equal(Ref,cphVB)

def test_jacoby():
    dtype = np.float32
    size = 50
    A = np.random.random((size,size), dtype=dtype, cphvb=False)
    b = np.random.random((size), dtype=dtype, cphvb=False)
    A += np.diag(np.add.reduce(A,-1)) #make sure A is diagonally dominant
    Ref = la.jacobi(A,b)
    A.cphvb=True
    b.cphvb=True
    cphVB = la.jacobi(A,b)
    cphVB.cphvb = False
    assert_almost_equal(Ref,cphVB,decimal=5)

def test_gameoflife():
    size = 50
    it = 50
    state = gameoflife.randomstate(size,size, cphvb=False)
    Ref = gameoflife.play(state, it)
    state.cphvb=True
    cphVB = gameoflife.play(state, it)
    cphVB.cphvb = False
    assert_array_equal(Ref,cphVB)

def test_jacobi_stencil():
    grid = jacobi_stencil.frezetrap(50,50,cphvb=False)
    Ref = jacobi_stencil.solve(grid)
    grid.cphvb=True
    cphVB = jacobi_stencil.solve(grid)
    cphVB.cphvb = False
    assert_array_equal(Ref,cphVB)

if __name__ == "__main__":
    run_module_suite()
