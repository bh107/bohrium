import bohrium as np
from numpytest import numpytest,gen_views,TYPES

class test_flatten(numpytest):
    def init(self):
        for v in gen_views(3,64,6):
            a = {}
            exec(v)
            yield (a,v)

    def test_flatten(self,a):
        cmd = "res = np.flatten(a[0])"
        exec(cmd)
        return (res,cmd)

    def test_flatten_self(self,a):
        cmd = "res = a[0].flatten()"
        exec(cmd)
        return (res,cmd)

    def test_ravel_self(self,a):
        cmd = "res = a[0].ravel()"
        exec(cmd)
        return (res,cmd)

class test_diagonal(numpytest):
    def init(self):
        for v in gen_views(4,32,6,min_ndim=2):
            a = {}
            exec(v)
            yield (a,v)

    def test_diagonal(self,a):
        cmd = "res = np.diagonal(a[0])"
        exec(cmd)
        return (res,cmd)

class test_diagonal_offset(numpytest):
    def init(self):
        for v in gen_views(4,16,6,min_ndim=2):
            a = {}
            self.offset = 0
            exec(v)
            yield (a,v)

            for offset in xrange(1, a[0].shape[0]):
                exec(v)
                self.offset = offset
                yield (a,v)

                exec(v)
                self.offset = -offset
                yield (a,v)

    def test_diagonal_offset(self,a):
        cmd = "res = np.diagonal(a[0], offset=%d)" % self.offset
        exec(cmd)
        return (res,cmd)

class test_diagonal_axis(numpytest):
    def init(self):
        for v in gen_views(4,16,6,min_ndim=2):
            a = {}
            self.axis1 = 0
            self.axis2 = 1
            exec(v)
            yield (a,v)

            for axis1 in xrange(a[0].ndim):
                for axis2 in xrange(a[0].ndim):
                    if axis1 == axis2:
                        continue
                    exec(v)
                    self.axis1 = axis1
                    self.axis2 = axis2
                    yield (a,v)

    def test_diagonal_axis(self,a):
        cmd = "res = np.diagonal(a[0], axis1=%d, axis2=%d)" % (self.axis1, self.axis2)
        exec(cmd)
        return (res,cmd)

class test_transpose(numpytest):
    def init(self):
        for v in gen_views(4,16,6,min_ndim=2):
            a = {}
            exec(v)
            yield (a,v)

    def test_transpose(self,a):
        cmd = "res = np.transpose(a[0])"
        exec(cmd)
        return (res,cmd)
