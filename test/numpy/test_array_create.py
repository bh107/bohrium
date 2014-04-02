import numpy as np
from numpytest import numpytest,TYPES

class test_array_create(numpytest):
    def init(self):
        for t in  TYPES.NORMAL:
            a = {}
            cmd = "a[0] = np.zeros(%d,dtype=%s)"%(100,t)
            exec cmd
            yield (a,cmd)

    def test_zeros(self,a):
        return (a[0],"")
