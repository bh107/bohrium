import numpy as np
import cphvbnumpy
import numpytest

dtype=np.float32

def f(DIST, SIZE):
    tmp1 = np.array(range(1,SIZE), dtype=dtype, cphvb=DIST)
    B    = np.array(range(1,SIZE), dtype=dtype, cphvb=DIST)
    AD   = np.array(range(1,SIZE), dtype=dtype, cphvb=DIST)
    for i in range(10):
        cphvbnumpy.flush()
        tmp1 += B
        tmp1 += AD
        cphvbnumpy.flush()
    return (tmp1,B,AD)

def run():
    res1 = f(True,98)
    res2 = f(False,98)

    for i in range(len(res1)):
        if not numpytest.array_equal(res1[i],res2[i]):
            raise Exception("Uncorrect result vector\n")

if __name__ == "__main__":
    run()


