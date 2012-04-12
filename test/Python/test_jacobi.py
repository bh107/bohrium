from numpy import *
import numpytest
import cphvbnumpy

def jacobi(A, B, tol=0.005):
    '''itteratively solving for matrix A with solution vector B
       tol = tolerance for dh/h
       init_val = array of initial values to use in the solver
    '''
    h = zeros(shape(B), float)
    cphvbnumpy.handle_array(h)
    dmax = 1.0
    n = 0
    tmp0 = empty(shape(A), float, dist=True)
    tmp1 = empty(shape(B), float, dist=True)
    AD = diagonal(A)
    while dmax > tol:
        n += 1
        multiply(A,h,tmp0)
        add.reduce(tmp0,1,out=tmp1)
        tmp2 = AD
        subtract(B, tmp1, tmp1)
        divide(tmp1, tmp2, tmp1)
        hnew = h + tmp1
        subtract(hnew,h,tmp2)
        if n != 1:
            divide(tmp2,h,tmp1)
            absolute(tmp1,tmp1)
            dmax = maximum.reduce(tmp1)
        else:
            dmax = 1
        h = hnew
    return h

def run():
    A = load("%sJacobi_Amatrix.npy"%numpytest.DataSetDir)
    B = load("%sJacobi_Bvector.npy"%numpytest.DataSetDir)
    C = load("%sJacobi_Cvector.npy"%numpytest.DataSetDir)

    cphvbnumpy.handle_array(A)
    cphvbnumpy.handle_array(B)
    result = jacobi(A,B)

    if not numpytest.array_equal(C,result):
        raise Exception("Uncorrect result vector\n")

if __name__ == "__main__":
    run()
