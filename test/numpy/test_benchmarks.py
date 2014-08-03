import os

from numpytest import numpytest, BenchHelper
import bohrium.linalg as la
import bohrium as bh

class test_gameoflife(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.size = 10

        # Benchmark parameters
        self.script     = "gameoflife"
        self.dtypes     = [bh.float64]
        self.sizetxt    = "10*10*50"
        self.inputfn    = "datasets/gameoflife_input-%s-12*12.npz"

    def test_gameoflife(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_shallow_water(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.size = 20

        # Benchmark parameters
        self.script     = "shallow_water"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "20*20*10"
        self.inputfn    = "datasets/shallow_water_input-%s-20*20.npz"

    def test_shallow_water(self, pseudo_arrays):
        return self.run(pseudo_arrays)

"""
class test_shallow_water(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.size = 20
    def init(self):
        for t in ['bh.float32','bh.float64']:
            a = {}
            cmd  = "a[0] = exp.shallow_water.model({0},{0},dtype={1});".format(self.size,t)
            exec cmd
            yield (a,cmd)

    def test_shallow_water(self,a):
        cmd = "res = exp.shallow_water.simulate(a[0],10);"
        exec cmd
        return (res,cmd)

class test_jacobi_stencil(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.001
        self.size = 20
    def init(self):
        a = {}
        cmd = "a[0] = exp.jacobi_stencil.freezetrap({0},{0});".format(self.size)
        exec cmd
        yield (a,cmd)

    def test_jacobi_stencil(self,a):
        cmd = "res = exp.jacobi_stencil.solve(a[0]);"
        exec cmd
        return (res,cmd)
"""

class test_jacobi(numpytest):#disabled
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.size = 20
    def init(self):
        print "We need to implement numpy.norm() for test_jacobi() to work"
        raise StopIteration()
        for t in ['bh.float32','bh.float64']:
            a = {}
            cmd  = "a[0] = self.array(({0},{0}),dtype={1});".format(self.size,t)
            cmd += "a[1] = self.array(({0}),dtype={1}); ".format(self.size,t)
            cmd += "a[0] += bh.diag(bh.add.reduce(a[0],-1));"
            exec cmd
            yield (a,cmd)

    def test_jacobi(self,a):
        cmd = "res = la.jacobi(a[0],a[1]);"
        exec cmd
        return (res,cmd)

