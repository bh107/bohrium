import numpy as np
import numpytest
import random

type=np.float32

def run():
    max_ndim = 6
    for i in xrange(1,max_ndim+1):
        print "[PyTest] Test 1. Dim: ", i
        src = numpytest.random_list(random.sample(xrange(1, 10),i))
        Ad = np.array(src, dtype=type, cphvb=True)
        Af = np.array(src, dtype=type, cphvb=False)
        ran = random.randint(0,i-1)
        if i > 1 and ran > 0:
            for j in range(0,ran):
                src = src[0]
        Bd = np.array(src, dtype=type, cphvb=True)
        Bf = np.array(src, dtype=type, cphvb=False)
        Cd = Ad + Bd + 42 + Bd[-1]
        Cf = Af + Bf + 42 + Bf[-1]
        Cd = Cd[::2] + Cd[::2,...] + Cd[0,np.newaxis]
        Cf = Cf[::2] + Cf[::2,...] + Cf[0,np.newaxis]
        Dd = np.array(Cd, dtype=type, cphvb=True)
        Df = np.array(Cf, dtype=type, cphvb=False)
        Dd[1:] = Cd[:-1]
        Df[1:] = Cf[:-1]
        Cd = Dd + Bd[np.newaxis,-1]
        Cf = Df + Bf[np.newaxis,-1]
        if not numpytest.array_equal(Cd,Cf):
            print "[PyTest] Test 1. Dim: ", i, ": Failed"
            raise Exception("Incorrect result array\n")
        print "[PyTest] Test 1. Dim: ", i, ": Passed"
    for i in xrange(3,max_ndim+3):
        print "[PyTest] Test 2. Dim: ", i
        src = numpytest.random_list([i,i,i])
        Ad = np.array(src, cphvb=True, dtype=type)
        Af = np.array(src, cphvb=False, dtype=type)
        Bd = np.array(src, cphvb=True, dtype=type)
        Bf = np.array(src, cphvb=False, dtype=type)
        Cd = Ad[::2, ::2, ::2] + Bd[::2, ::2, ::2] + Ad[::2,1,2]
        Cf = Af[::2, ::2, ::2] + Bf[::2, ::2, ::2] + Af[::2,1,2]
        if not numpytest.array_equal(Cd,Cf):
            print "[PyTest] Test 2. Dim: ", i, ": Failed"
            raise Exception("Incorrect result array\n")
        print "[PyTest] Test 2. Dim: ", i, ": Passed"


if __name__ == "__main__":
    run()
