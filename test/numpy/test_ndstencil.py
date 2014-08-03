import bohrium as bh
from numpytest import numpytest, BenchHelper

class test_ndstencil_1D(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.size = 10

        # Benchmark parameters
        self.script     = "ndstencil"
        self.dtypes     = [bh.float64]
        self.sizetxt    = "10*100*1"
        self.inputfn    = "datasets/ndstencil_input-%s-1026.npz"

    def test_ndstencil_1D(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_ndstencil_2D(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.size = 10

        # Benchmark parameters
        self.script     = "ndstencil"
        self.dtypes     = [bh.float64]
        self.sizetxt    = "10*100*2"
        self.inputfn    = "datasets/ndstencil_input-%s-34*34.npz"

    def test_ndstencil_2D(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_ndstencil_3D(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.size = 10

        # Benchmark parameters
        self.script     = "ndstencil"
        self.dtypes     = [bh.float64]
        self.sizetxt    = "10*100*3"
        self.inputfn    = "datasets/ndstencil_input-%s-10*10*18.npz"

    def test_ndstencil_3D(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_ndstencil_4D(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.size = 10

        # Benchmark parameters
        self.script     = "ndstencil"
        self.dtypes     = [bh.float64]
        self.sizetxt    = "10*100*1"
        self.inputfn    = "datasets/ndstencil_input-%s-6*6*10*10.npz"

    def test_ndstencil_4D(self, pseudo_arrays):
        return self.run(pseudo_arrays)

""":q

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
    """
