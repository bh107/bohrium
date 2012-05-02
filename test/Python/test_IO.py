import numpy as np
import numpytest
import random
import subprocess
import cphvbnumpy

def run():
    try:#This test requires zlib
        import zlib
    except:
        print "Warning - ignored zlib not found\n",
        return

    max_ndim = 6
    for i in xrange(1,max_ndim+1):
        src = numpytest.random_list(random.sample(xrange(1, 10),i))
        A = np.array(src, dtype=float, cphvb=True)
        fname = "distnumpt_test_matrix.npy"
        np.save(fname,A)
        B = np.load(fname)

        if not numpytest.array_equal(A,B):
            subprocess.check_call(('rm %s'%fname), shell=True)
            raise Exception("Uncorrect result array\n")
        subprocess.check_call(('rm %s'%fname), shell=True)

if __name__ == "__main__":
    run()
