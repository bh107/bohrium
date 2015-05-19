import bohrium as np
from numpytest import numpytest, TYPES

class test_sign(numpytest):

    def init(self):
        self.config['maxerror'] = 0.00001

        for dtype in TYPES.ALL_SIGNED:
            a = {}
            cmd = "a[0] = self.arange(-10, 0, 1, dtype=%s);" % dtype
            exec(cmd)
            yield (a, cmd)

    def test_sign(self,a):
        cmd = "res = np.sign(a[0])"
        exec(cmd)
        return (res, cmd)

class test_csign_neg(numpytest):

    def init(self):
        self.config['maxerror'] = 0.00001

        for dtype in TYPES.COMPLEX:
            a = {}
            cmd = "a[0] = self.arange(-10, 1, 1, dtype=%s);" % dtype
            exec(cmd)
            yield (a, cmd)

    def test_sign(self,a):
        cmd = "res = np.sign(a[0])"
        exec(cmd)
        return (res, cmd)

class test_csign_pos(numpytest):

    def init(self):
        self.config['maxerror'] = 0.00001

        for dtype in TYPES.COMPLEX:
            a = {}
            cmd = "a[0] = self.arange(1, 10, 1, dtype=%s);" % dtype
            exec(cmd)
            yield (a, cmd)

    def test_sign(self,a):
        cmd = "res = np.sign(a[0])"
        exec(cmd)
        return (res, cmd)

class test_csign_zero(numpytest):

    def init(self):
        self.config['maxerror'] = 0.00001

        for dtype in TYPES.COMPLEX:
            a = {}
            cmd = "a[0] = self.zeros((10), dtype=%s);" % dtype
            exec(cmd)
            yield (a, cmd)

    def test_sign(self,a):
        cmd = "res = np.sign(a[0])"
        exec(cmd)
        return (res, cmd)

class test_csign_mixed(numpytest):

    def init(self):
        self.config['maxerror'] = 0.00001

        signs = []
        for x in xrange(-1,1):
            for y in xrange(-1,1):
                exec("z = %d+%dj"% (x,y))
                signs.append([z]*10)

        for dtype in TYPES.COMPLEX:
            for sign in signs:
                a = {}
                cmd = "a[0] = self.asarray(%s, dtype=%s);" % (sign, dtype)
                exec(cmd)
                yield (a, cmd) 

    def test_sign(self,a):
        cmd = "res = np.sign(a[0])"
        exec(cmd)
        return (res, cmd)
