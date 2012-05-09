import numpy as np
import cphvbbridge as cnp
import util
import sys
sys.path.append('../programs/')
from jacobi import jacobi

dtype=np.float32

def run():

    A = np.random.random((400,400), dtype=dtype)
    A += np.diag(np.add.reduce(A)) #make sure A is diagonally dominant
    b = np.random.random(400, dtype=dtype)
    
    Ref = jacobi(A,b)

    cnp.handle_array(A)
    cnp.handle_array(b)
    cphVB = jacobi(A,b)
    cnp.unhandle_array(cphVB)

    print Ref
    print cphVB

    if not util.array_equal(Ref,cphVB,maxerror=0.0):
        raise Exception("Incorrect result vector\n")

if __name__ == "__main__":
    run()
