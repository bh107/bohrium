import numpy as np
from numpytest import numpytest,gen_views,TYPES


class test_reduce(numpytest):
    
    def init(self):
        for v in gen_views(5,10,6):
            a = {}
            self.axis = 0
            exec v
            yield (a,v)
            for axis in xrange(1,a[0].ndim):
                exec v
                self.axis = axis
                yield (a,v)
                
    def test_reduce(self,a):
        cmd = "res = np.add.reduce(a[0],axis=%d)"%self.axis
        exec cmd
        return (res,cmd)

