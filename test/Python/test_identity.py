import numpy as np
import numpytest
import random

def run():
    max_ndim = 4
    for i in range(1,max_ndim+1):
        src = numpytest.random_list(random.sample(range(3, 32),i))
        Ad = np.array(src, dtype=float, cphvb=True)
        Af = np.array(src, dtype=float, cphvb=False)
        Ad[1:] = Ad[:-1]
        Af[1:] = Af[:-1]
        np.add(Ad[:-1], 42, Ad[1:]);
        np.add(Af[:-1], 42, Af[1:]);
        if not numpytest.array_equal(Ad,Af):
            raise Exception("Uncorrect result array\n")
    return (False, "")

if __name__ == "__main__":
    run()
