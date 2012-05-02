import numpy as np
import numpytest
import random

def run():
    niter = 6
    for m in range(2,niter+2):
        for n in range(2,niter+2):
            for k in range(2,niter+2):
                Asrc = numpytest.random_list([k,m])
                Bsrc = numpytest.random_list([m,k])
                Ad = np.array(Asrc, dtype=float, cphvb=True)
                Af = np.array(Asrc, dtype=float, cphvb=False)
                Bd = np.array(Bsrc, dtype=float, cphvb=True)
                Bf = np.array(Bsrc, dtype=float, cphvb=False)
                Cd = np.dot(Ad,Bd)
                Cf = np.dot(Af,Bf)
                if not numpytest.array_equal(Cd,Cf):
                    raise Exception("Uncorrect result matrix\n")

if __name__ == "__main__":
    run()
