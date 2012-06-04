import numpy as np
import cphvbnumpy as cnp
from numpy.testing import assert_array_equal, assert_, run_module_suite
from numpy.random import randint, random

def test_zeros():
    n = randint(2,100)
    assert_array_equal(np.zeros(n),cnp.zeros(n,cphvb=False))

def test_ones():
    n = randint(2,100)
    assert_array_equal(np.ones(n),cnp.ones(n,cphvb=False))

def test_flatten():
    A = random(randint(5,15,size=randint(2,8)))
    assert_array_equal(A.flatten(),cnp.flatten(A))

def test_diagonal():
    n = randint(50,100)
    A = random((n,n))
    k = randint(10,20)
    assert_array_equal(np.diagonal(A),cnp.diagonal(A))
    assert_array_equal(np.diagonal(A,offset=k),cnp.diagonal(A,offset=k))
    assert_array_equal(np.diagonal(A,offset=-k),cnp.diagonal(A,offset=-k))
    m = n + randint(-20,20)
    print n,m
    A = random((n,m))
    assert_array_equal(np.diagonal(A),cnp.diagonal(A))
    assert_array_equal(np.diagonal(A,offset=k),cnp.diagonal(A,offset=k))
    assert_array_equal(np.diagonal(A,offset=-k),cnp.diagonal(A,offset=-k))

def test_matmul():
    niter = 6
    for m in range(2,niter+2):
        for n in range(2,niter+2):
            for k in range(2,niter+2):
                A = random((k,m))
                B = random((m,k))
                C1 = np.dot(A,B)
                A.cphvb = True
                B.cphvb = True
                C2 = cnp.matmul(A,B)
                C2.cphvb = False
                assert_array_equal(C1,C2)
                print C1
                print C2
if __name__ == "__main__":
    run_module_suite()
