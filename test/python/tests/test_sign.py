import util

class test_sign:

    def init(self):
        for dtype in util.TYPES.SIGNED_INT:
            yield "a = M.arange(-10, 0, 1, dtype=%s); " % dtype

    def test_sign(self, cmd):
        return cmd + "res = M.sign(a)"

"""
class test_csign:

    def init(self):
        for dtype in util.TYPES.COMPLEX:
            yield "a = M.arange(-10, 0, 1, dtype=%s); " % dtype

    def test_sign(self, cmd):
        return cmd + "res = M.sign(a)"

class test_csign_neg(numpytest):

    def init(self):
        self.config['maxerror'] = 0.00001

        for dtype in TYPES.COMPLEX:
            a = {}
            cmd = "a[0] = self.arange(-10, 0, 1, dtype=%s);" % dtype
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
        for x in xrange(-1,2):
            for y in xrange(-1,2):
                exec("z = %d+%dj"% (x,y))
                signs.append(z)

        for dtype in TYPES.COMPLEX:
            a = {}
            cmd = "a[0] = self.asarray(%s, dtype=%s);" % (signs, dtype)
            exec(cmd)
            yield (a, cmd)

    def test_sign(self,a):
        cmd = sign_str+"res = sign(a[0])"
        exec(cmd)
        return (res, cmd)
"""
