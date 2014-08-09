import os

from numpytest import numpytest, BenchHelper
import bohrium.linalg as la
import bohrium as bh

#
#   Testing benchmarks via benchmark scripts
#
class test_black_scholes(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size = 10000

        # Benchmark parameters
        self.script     = "black_scholes"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "10000*10"
        self.inputfn    = "black_scholes_input-{0}-10000.npz"

    def test_black_scholes(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_gameoflife(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.size = 10

        # Benchmark parameters
        self.script     = "gameoflife"
        self.dtypes     = [bh.float64]
        self.sizetxt    = "10*10*50"
        self.inputfn    = "gameoflife_input-{0}-12*12.npz"

    def test_gameoflife(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_lu(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.size = 100

        # Benchmark parameters
        self.script     = "lu"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "100*100"
        self.inputfn    = "lu_input-{0}-100*100.npz"

    def test_lu(self, pseudo_arrays):
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
        self.inputfn    = "shallow_water_input-{0}-20*20.npz"

    def test_shallow_water(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_sor(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.size = 1000

        # Benchmark parameters
        self.script     = "sor"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "1000*1000*10"
        self.inputfn    = None

    def test_sor(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_heat_equation(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size = 1000

        # Benchmark parameters
        self.script     = "heat_equation"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "1000*1000*10"
        self.inputfn    = None

    def test_heat_equation(self, pseudo_arrays):
        return self.run(pseudo_arrays)

""" Segfaults so we cannot test it.
class test_jacobi(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size = 2

        # Benchmark parameters
        self.script     = "jacobi"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "2"
        self.inputfn    = None

    def test_jacobi(self, pseudo_arrays):
        return self.run(pseudo_arrays)
"""

class test_jacobi_fixed(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size = 1000

        # Benchmark parameters
        self.script     = "jacobi_fixed"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "1000*10"
        self.inputfn    = None

    def test_jacobi_fixed(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_jacobi_solve(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.001
        self.size = 20

        # Benchmark parameters
        self.script     = "jacobi_solve"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "1000*1000*10"
        self.inputfn    = None

    def test_jacobi_solve(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_jacobi_stencil(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.001
        self.size = 20

        # Benchmark parameters
        self.script     = "jacobi_stencil"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "1000*1000*10"
        self.inputfn    = None

    def test_jacobi_stencil(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_knn_naive(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size=10000

        # Benchmark parameters
        self.script     = "knn.naive"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "10000*100*3"
        self.inputfn    = None

class test_snakes_and_ladders(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size = 100

        # Benchmark parameters
        self.script     = "snakes_and_ladders"
        self.dtypes     = [bh.float64]
        self.sizetxt    = "100*100"
        self.inputfn    = "snakes_and_ladders_a-{0}-101_p-{0}-101*101.npz"

    def test_snakes_and_ladders(self, pseudo_arrays):
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
        self.inputfn    = "gauss_input-{0}-20*20.npz"

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
        self.inputfn    = "wireworld_input-{0}-1002*1002.npz"

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
        self.inputfn    = "mxmul_y-{0}-500*500_x-{0}-500*500.npz"

    def test_mxmul(self, pseudo_arrays):
        return self.run(pseudo_arrays)

""" Cannot run this since it breaks due to a futureWarning.
class test_nbody(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size = 100

        # Benchmark parameters
        self.script     = "nbody"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "100*100"
        self.inputfn    = "nbody_m-{0}-100_vx-{0}-100_vy-{0}-100_y-{0}-100_x-{0}-100_vz-{0}-100_z-{0}-100.npz"

    def test_nbody(self, pseudo_arrays):
        return self.run(pseudo_arrays)
"""

#
#   Testing via import of modules
#
class test_jacobi_module(numpytest):#disabled
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

    def test_jacobi_module(self,a):
        cmd = "res = la.jacobi(a[0],a[1]);"
        exec cmd
        return (res,cmd)
