import bohrium as np
from numpytest import numpytest
from bohrium import examples as exp

class test_ndstencil(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
    def init(self):
        for dim in xrange (1,5):
            a = {}
            self.size = (dim,10,100)
            cmd = "a[0] = exp.ndstencil.worldND({0},{1},dtype=np.float32,bohrium=False);".format(*self.size)
            exec cmd
            yield (a,cmd)

    def test_ndstencil(self,a):
        cmd = "res = exp.ndstencil.solveND(a[0],{2});".format(*self.size)
        exec cmd
        return (res,cmd)
