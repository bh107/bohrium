import numpy as np
from numpytest import numpytest,gen_views,TYPES


class test_accumulate(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
    
    def init(self):
        for v in gen_views(4,10,6,min_ndim=2):
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


class test_accumulate1D(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
    
    def init(self):
        for v in gen_views(1,100,10):
            a = {}
            v += "a[0] = a[0][:, np.newaxis];"
            exec v
            for l in xrange(len(a[0])-1):
                v2 = v + "a[1] = self.array([100], np.%s);"%(a[0].dtype)
                v2 += "a[2] = a[1][%d:%d];"%(l, l+1)
                exec v2
                yield (a,v2)
                
    def test_cumsum(self,a):
        cmd = "np.cumsum(a[0], out=a[2])"
        exec cmd
        return (a[1],cmd)

    def test_cumprod(self,a):
        cmd = "np.cumprod(a[0], out=a[2])"
        exec cmd
        return (a[1],cmd)

