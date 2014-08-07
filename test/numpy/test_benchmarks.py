import os

from numpytest import numpytest, BenchHelper
import bohrium.linalg as la
import bohrium as bh

#
#   Testing benchmarks via benchmark scripts
#
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

class test_jacobi_solve(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.001
        self.size = 20

        # Benchmark parameters
        self.script     = "jacobi_solve"
        self.dtypes     = [bh.float64]
        self.sizetxt    = "20*20*10"
        self.inputfn    = "datasets/jacobi_solve_input-%s-22*22.npz"

    def test_jacobi_solve(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_gauss(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size = 20

        # Benchmark parameters
        self.script     = "gauss"
        self.dtypes     = [bh.float64]
        self.sizetxt    = "20*20"
        self.inputfn    = "datasets/gauss_input-%s-20*20.npz"

    def test_gauss(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_wireworld(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size = 100

        # Benchmark parameters
        self.script     = "wireworld"
        self.dtypes     = [bh.uint8]
        self.sizetxt    = "100*10"
        self.inputfn    = "datasets/wireworld_input-%s-1002*1002.npz"

    def test_wireworld(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_mxmul(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size = 500

        # Benchmark parameters
        self.script     = "mxmul"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "500"
        self.inputfn    = "datasets/mxmul_y-{0}-500*500_x-{0}-500*500.npz"

    def test_mxmul(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_nbody(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size = 100

        # Benchmark parameters
        self.script     = "nbody"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "100*100"
        self.inputfn    = "datasets/nbody_m-{0}-100_vx-{0}-100_vy-{0}-100_y-{0}-100_x-{0}-100_vz-{0}-100_z-{0}-100.npz"

    def test_nbody(self, pseudo_arrays):
        return self.run(pseudo_arrays)

#
#   Testing via import of modules
#
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
