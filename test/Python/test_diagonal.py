import numpy as np
import numpytest
import random

def run():
    niters = 10
    for i in xrange(niters):
        for j in xrange(niters):
            src = numpytest.random_list([i+1,j+1])
            Ad = np.array(src, dtype=float, cphvb=True)
            Af = np.array(src, dtype=float, cphvb=False)
            Cd = Ad.diagonal()
            Cf = Af.diagonal()
            if not numpytest.array_equal(Cd,Cf):
                raise Exception("Uncorrect result matrix\n")

if __name__ == "__main__":
    run()
