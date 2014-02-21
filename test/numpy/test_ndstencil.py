import bohrium as np
from numpytest import numpytest
import bohrium.examples.ndstencil as nds

class test_ndstencil(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
    def init(self):
        for dim in xrange (1,5):
            a = {}
            self.size = (dim,10,100)
            cmd = "a[0] = np.random.random(nds.shape({0},{1}),bohrium=False);".format(*self.size)
            exec cmd
            yield (a,cmd)

    def test_ndstencil(self,a):
        cmd = "res = nds.solve(a[0],{2});".format(*self.size)
        exec cmd
        return (res,cmd)
