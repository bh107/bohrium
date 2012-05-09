from numpy import *
import numpytest
import cphvbbridge

def jacobi(A, B, tol=0.05):
    '''itteratively solving for matrix A with solution vector B
       tol = tolerance for dh/h
       init_val = array of initial values to use in the solver
    '''
    h = zeros(shape(B), float)
    cphvbbridge.handle_array(h)
    dmax = 1.0
    n = 0
    AD = diagonal(A)
    cphvbbridge.handle_array(AD)
    while dmax > tol:
        n += 1
        t = add.reduce(A * h, axis=1)
        t -= B
        t /= AD
        hnew = h + t
        AD = hnew - h
        if n != 1:
            t = absolute(AD / h)
            dmax = maximum.reduce(t)
        else:
            dmax = 1.0
        h = hnew
    return h

def run():
    A = load("%sJacobi_Amatrix.npy"%numpytest.DataSetDir)
    B = load("%sJacobi_Bvector.npy"%numpytest.DataSetDir)
    cphvbbridge.handle_array(A)
    cphvbbridge.handle_array(B)
    result1 = jacobi(A,B)

    A = load("%sJacobi_Amatrix.npy"%numpytest.DataSetDir)
    B = load("%sJacobi_Bvector.npy"%numpytest.DataSetDir)
    result2 = jacobi(A,B)

    if not numpytest.array_equal(result1,result2):
        raise Exception("Uncorrect result vector\n")

if __name__ == "__main__":
    run()
