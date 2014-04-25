import numpy as np
from numpytest import numpytest,gen_views,TYPES

class test_accumulate(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001

    def init(self):
        print "test_accumulate isn't support by the new bridge"
        raise StopIteration
        for v in gen_views(4,10,6,min_ndim=1):
            a = {}
            self.axis = 0
            exec v
            yield (a,v)
            for axis in xrange(1,a[0].ndim):
                exec v
                self.axis = axis
                yield (a,v)

    def test_cumsum(self,a):
        cmd = "res = np.cumsum(a[0],axis=%d)"%self.axis
        exec cmd
        return (res,cmd)

    def test_cumprod(self,a):
        cmd = "res = np.cumprod(a[0],axis=%d)"%self.axis
        exec cmd
        return (res,cmd)

