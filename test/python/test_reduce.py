import bohrium as np
import numpy
from numpytest import numpytest,gen_views,TYPES

class test_reduce(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001

    def init(self):
        for v in gen_views(4,10,6):
            a = {}
            self.axis = 0
            exec(v)
            yield (a,v)
            for axis in range(1,a[0].ndim):
                exec(v)
                self.axis = axis
                yield (a,v)
            for axis in range(1,a[0].ndim):
                exec(v)
                self.axis = -axis
                yield (a,v)

    def test_reduce_add(self,a):
        cmd = "res = np.add.reduce(a[0],axis=%d)"%self.axis
        exec(cmd)
        return (res,cmd)

    def test_reduce_mul(self,a):
        cmd = "res = np.multiply.reduce(a[0],axis=%d)"%self.axis
        exec(cmd)
        return (res,cmd)

    def test_reduce_min(self,a):
        cmd = "res = np.minimum.reduce(a[0],axis=%d)"%self.axis
        exec(cmd)
        return (res,cmd)

    def test_reduce_max(self,a):
        cmd = "res = np.maximum.reduce(a[0],axis=%d)"%self.axis
        exec(cmd)
        return (res,cmd)

class test_reduce_bitwise(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001

    def init(self):
        for v in gen_views(4,10,6, dtype="np.uint32"):
            a = {}
            self.axis = 0
            exec(v)
            yield (a,v)
            for axis in range(1,a[0].ndim):
                exec(v)
                self.axis = axis
                yield (a,v)
            for axis in range(1,a[0].ndim):
                exec(v)
                self.axis = -axis
                yield (a,v)

    def test_reduce_bitwise_and(self,a):
        cmd = "res = np.bitwise_and.reduce(a[0],axis=%d)"%self.axis
        exec(cmd)
        return (res,cmd)

    def test_reduce_bitwise_or(self,a):
        cmd = "res = np.bitwise_or.reduce(a[0],axis=%d)"%self.axis
        exec(cmd)
        return (res,cmd)

    def test_reduce_bitwise_xor(self,a):
        cmd = "res = np.bitwise_xor.reduce(a[0],axis=%d)"%self.axis
        exec(cmd)
        return (res,cmd)

class test_reduce_bool(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001

    def init(self):
        for v in gen_views(4,10,6,dtype="np.bool"):
            a = {}
            self.axis = 0
            exec(v)
            yield (a,v)
            for axis in range(1,a[0].ndim):
                exec(v)
                self.axis = axis
                yield (a,v)
            for axis in range(1,a[0].ndim):
                exec(v)
                self.axis = -axis
                yield (a,v)

    def test_boolean(self,a):
        cmd = "res = np.add.reduce(a[0],axis=%d)"%self.axis
        exec(cmd)
        return (res,cmd)

    def test_logical_or(self,a):
        cmd = "res = np.logical_or.reduce(a[0],axis=%d)"%self.axis
        exec(cmd)
        return (res,cmd)

    def test_logical_and(self,a):
        cmd = "res = np.logical_and.reduce(a[0],axis=%d)"%self.axis
        exec(cmd)
        return (res,cmd)

    def test_logical_xor(self,a):
        cmd = "res = np.logical_xor.reduce(a[0],axis=%d)"%self.axis
        exec(cmd)
        return (res,cmd)


class test_reduce_sum(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001

    def init(self):
        for v in gen_views(4,10,6):
            a = {}
            self.axis = 0
            exec(v)
            yield (a,v)

    def test_add_reduce(self,a):
        cmd = "res = np.sum(a[0])"
        exec(cmd)
        return (res,cmd)

    def test_sum(self,a):
        cmd = "res = a[0].sum()"
        exec(cmd)
        return (res,cmd)

class test_reduce_prod(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001

    def init(self):
        for v in gen_views(4,10,6):
            a = {}
            self.axis = 0
            exec(v)
            yield (a,v)

    def test_mul_reduce(self,a):
        cmd = "res = np.prod(a[0])"
        exec(cmd)
        return (res,cmd)

    def test_prod(self,a):
        cmd = "res = a[0].prod()"
        exec(cmd)
        return (res,cmd)

class test_reduce1D(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001

    def init(self):
        for v in gen_views(1,100,10):
            a = {}
            v += "a[0] = a[0][:, np.newaxis];"
            exec(v)
            for l in range(len(a[0])-1):
                self.l = l
                v2 = v + "a[1] = self.array([100], np.%s);"%(a[0].dtype)
                exec(v2)
                yield (a,v2)

    def test_reduce(self,a):
        cmd = "t2 = a[1][%d:%d];"%(self.l, self.l+1)
        cmd += "np.add.reduce(a[0], out=t2)"
        exec(cmd)
        return (a[1],cmd)

