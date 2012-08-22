import cphvbnumpy as np
from numpytest import numpytest
import cphvbnumpy.linalg as la
from cphvbnumpy import examples as exp

class test_jacobi(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size = 20
    def init(self):
        for t in ['np.float32','np.float64']:
            a = {}
            cmd  = "a[0] = self.array(({0},{0}),dtype={1});".format(self.size,t)
            cmd += "a[1] = self.array(({0}),dtype={1}); ".format(self.size,t)
            cmd += "a[0] += np.diag(np.add.reduce(a[0],-1));"
            exec cmd
            yield (a,cmd)
    
    def test_jacobi(self,a):
        cmd = "res = la.jacobi(a[0],a[1]);"
        exec cmd
        return (res,cmd)

class test_jacobi_stencil(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.01
        self.size = 20
    def init(self):
        a = {}
        cmd = "a[0] = exp.jacobi_stencil.frezetrap({0},{0},cphvb=False);".format(self.size)
        exec cmd
        yield (a,cmd)

    def test_jacobi_stencil(self,a):
        cmd = "res = exp.jacobi_stencil.solve(a[0]);"
        exec cmd
        return (res,cmd)

class test_gameoflife(numpytest):
    def init(self):
        a = {}
        cmd  = "a[0] = exp.gameoflife.randomstate({0},{0},cphvb=False);".format(10)
        exec cmd
        yield (a,cmd)
    
    def test_gameoflife(self,a):
        cmd = "res = exp.gameoflife.play(a[0].copy(), 50);"
        exec cmd
        return (res,cmd)

class test_shallow_water(numpytest):
    def init(self):
        for t in ['np.float32','np.float64']:
            a = {}
            cmd  = "a[0] = exp.shallow_water.setup(({0},{0}),dtype={1},cphvb=False);".format(10,t)
            exec cmd
            yield (a,cmd)
    
    def test_shallow_water(self,a):
        cmd = "res = exp.shallow_water.simulate(a[0],10);"
        exec cmd
        return (res,cmd)

