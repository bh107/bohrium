import numpy as np
import cphvbnumpy as cnp
import util
import sys
sys.path.append('../programs/')
from jacobi import jacobi

dtype=np.float32

def run():

    A = np.random.random((50,50), dtype=dtype)
    A += np.diag(np.add.reduce(A)) #make sure A is diagonally dominant
    b = np.random.random(50, dtype=dtype)
    
    Ref = jacoby(A,b)

    cnp.handle_array(A)
    cnp.handle_array(b)
    cphVB = solve(A,b)
    cnp.unhandle_array(cphVB)

    if not util.array_equal(REf,cphVB):
        raise Exception("Incorrect result vector\n")

if __name__ == "__main__":
    run()
