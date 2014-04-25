import bohrium as np
from numpytest import numpytest,gen_views,TYPES


class test_reduce(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001

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


class test_reduce1D(numpytest):
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

    def test_reduce(self,a):
        cmd = "np.add.reduce(a[0], out=a[2])"
        exec cmd
        return (a[1],cmd)

