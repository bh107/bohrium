import numpy as np
import numpytest
import cphvbnumpy as cnp

def jacobi(A, B, cphvb, tol=0.005):
    '''itteratively solving for matrix A with solution vector B
       tol = tolerance for dh/h
       init_val = array of initial values to use in the solver
    '''
    h = np.zeros(np.shape(B), np.float32)
    if cphvb:
        cnp.handle_array(A)
        cnp.handle_array(B)
        cnp.handle_array(h)
    dmax = 1.0
    n = 0
    tmp0 = np.empty(np.shape(A), np.float32, cphvb=cphvb)
    tmp1 = np.empty(np.shape(B), np.float32, cphvb=cphvb)
    AD = np.diagonal(A)
    while dmax > tol:
        n += 1
        np.multiply(A,h,tmp0)
        np.add.reduce(tmp0,1,out=tmp1)
        tmp2 = AD
        np.subtract(B, tmp1, tmp1)
        np.divide(tmp1, tmp2, tmp1)
        hnew = h + tmp1
        np.subtract(hnew,h,tmp2)
        if n != 1:
            np.divide(tmp2,h,tmp1)
            np.absolute(tmp1,tmp1)
            dmax = np.maximum.reduce(tmp1)
        else:
            dmax = 1
        h = hnew
    return h

def run():
    A = np.load("%sJacobi_Amatrix.npy"%numpytest.DataSetDir)
    B = np.load("%sJacobi_Bvector.npy"%numpytest.DataSetDir)

    resNPY = jacobi(A,B,False)
    resCPHVB = jacobi(A,B,True)

    if not numpytest.array_equal(resCPHVB,resNPY):
        raise Exception("Incorrect result vector\n")

if __name__ == "__main__":
    run()
