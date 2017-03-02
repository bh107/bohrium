from __future__ import print_function
import os

from numpytest import numpytest, BenchHelper
import bohrium.linalg as la
import bohrium as bh

#
#   Testing benchmarks via benchmark scripts
#

"""
Cannot test:

    LMM_swaption_vec.py

Due to missing functionality: concatenate.
"""

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

class test_rosenbrock(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size = 100000

        # Benchmark parameters
        self.script     = "rosenbrock"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "10000*10"
        self.inputfn    = "rosenbrock_input-{0}-100000.npz"

    def test_rosenbrock(self, pseudo_arrays):
        return self.run(pseudo_arrays)

"""
Cannot test:

    convolve.py
    convolve_2d.py
    convolve_3d.py
    convolve_separate_std.py

Due to errors.
"""

class test_gameoflife(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.size = 10

        # Benchmark parameters
        self.script     = "gameoflife"
        self.dtypes     = [bh.float64]
        self.sizetxt    = "10*10*50*1"
        self.inputfn    = "gameoflife_input-{0}-12x12.npz"

    def test_gameoflife(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_gauss(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size = 4

        # Benchmark parameters
        self.script     = "gauss"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "4*4"
        self.inputfn    = "gauss_input-{0}-4x4.npz"

    def test_gauss(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_gauss_20x20(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.002
        self.size = 20

        # Benchmark parameters
        self.script     = "gauss"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "20*20"
        self.inputfn    = "gauss_input-{0}-20x20.npz"

    def test_gauss_20x20(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_heat_equation(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.001
        self.size = 20

        # Benchmark parameters
        self.script     = "heat_equation"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "1000*1000*10"
        self.inputfn    = None

    def test_heat_equation(self, pseudo_arrays):
        return self.run(pseudo_arrays)


"""
Cannot test: k_nearest_neighbor.py

It needs to have main() implemented.
"""

class test_knn_naive(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.01
        self.size=10000

        # Benchmark parameters
        self.script     = "knn_naive"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "10000*100*3"
        self.inputfn    = None

    def test_knn_naive(self, pseudo_arrays):
        return self.run(pseudo_arrays)

"""
Cannot test: lattice_boltzmann_D2Q9.py

    Traceback (most recent call last):
      File "benchmark/Python/lattice_boltzmann_D2Q9.py", line 197, in <module>
        cylinder = cylinder(H, W, obstacle=False)
      File "benchmark/Python/lattice_boltzmann_D2Q9.py", line 42, in cylinder
        t_3d    = np.asarray(t)[:, np.newaxis, np.newaxis]
    AttributeError: 'module' object has no attribute 'asarray'
"""

"""
Cannot test: lbm_3d.py

Test breaks due to overflow errors.
class test_lbm_3d(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size = 10

        # Benchmark parameters
        self.script     = "lbm_3d"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "5*5*5*10"
        self.inputfn    = None

    def test_lbm_3d(self, pseudo_arrays):
        return self.run(pseudo_arrays)
"""

class test_lu(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.01
        self.size = 100

        # Benchmark parameters
        self.script     = "lu"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "100*100"
        self.inputfn    = "lu_input-{0}-100x100.npz"

    def test_lu(self, pseudo_arrays):
        return self.run(pseudo_arrays)

"""
Cannot test: mc.py

Since it does random stuff in each iteration...
"""

class test_mxmul(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size = 500

        # Benchmark parameters
        self.script     = "mxmul"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "500"
        self.inputfn    = "mxmul_y-{0}-500x500_x-{0}-500x500.npz"

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
        self.inputfn    = "nbody_m-{0}-100_vx-{0}-100_vy-{0}-100_y-{0}-100_x-{0}-100_vz-{0}-100_z-{0}-100.npz"

    def test_nbody(self, pseudo_arrays):
        return self.run(pseudo_arrays)

"""
See test_ndstencil.py for testing for ndstencil.py
"""

class test_point27(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size=100

        # Benchmark parameters
        self.script     = "point27"
        self.dtypes     = [bh.float64]
        self.sizetxt    = "100*10"
        self.inputfn    = None

    def test_point27(self, pseudo_arrays):
        return self.run(pseudo_arrays)

"""
Cannot test: pricing.py

It segfaults when running with Bohrium.
"""

class test_shallow_water(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.size = 20

        # Benchmark parameters
        self.script     = "shallow_water"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "20*20*10"
        self.inputfn    = "shallow_water_input-{0}-20x20.npz"

    def test_shallow_water(self, pseudo_arrays):
        return self.run(pseudo_arrays)


class test_snakes_and_ladders(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size = 100

        # Benchmark parameters
        self.script     = "snakes_and_ladders"
        self.dtypes     = [bh.float64]
        self.sizetxt    = "100*100"
        self.inputfn    = "snakes_and_ladders_a-{0}-101_p-{0}-101x101.npz"

    def test_snakes_and_ladders(self, pseudo_arrays):
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

class test_synth(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.size = 1000

        # Benchmark parameters
        self.script     = "synth"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "1000*100"
        self.inputfn    = None

    def test_synth(self, pseudo_arrays):
        return self.run(pseudo_arrays)

"""
Works with input-files but not without due to missing functionality:

    Traceback (most recent call last):
      File "benchmark/Python/wireworld.py", line 55, in <module>
        world = wireworld_init(N)
      File "benchmark/Python/wireworld.py", line 15, in wireworld_init
        data[1:-1,1:-1] = np.tile(np.array([
    AttributeError: 'module' object has no attribute 'tile'
"""
class test_wireworld(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size = 100

        # Benchmark parameters
        self.script     = "wireworld"
        self.dtypes     = [bh.uint8]
        self.sizetxt    = "100*10"
        self.inputfn    = "wireworld_input-{0}-1002x1002.npz"

    def test_wireworld(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_mfe(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.0001
        self.size = 16

        # Benchmark parameters
        self.script     = "magnetic_field_extrapolation"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "16*16*5"
        self.inputfn    = "idl_input-float64_512x512.npz"

    def test_mfe(self, pseudo_arrays):
        return self.run(pseudo_arrays)

class test_xraysim(BenchHelper, numpytest):

    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.size = 32

        # Benchmark parameters
        self.script     = "xraysim"
        self.dtypes     = [bh.float32, bh.float64]
        self.sizetxt    = "32*10*2"
        self.inputfn    = None

    def test_xraysim(self, pseudo_arrays):
        return self.run(pseudo_arrays)

#
#   Testing via import of modules
#
class test_jacobi_module(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.size = 20
    def init(self):
        for t in ['bh.float32','bh.float64']:
            a = {}
            cmd  = "a[0] = self.array(({0},{0}),dtype={1});".format(self.size,t)
            cmd += "a[1] = self.array(({0}),dtype={1}); ".format(self.size,t)
            cmd += "a[0] += bh.diag(bh.add.reduce(a[0],-1));"
            exec(cmd)
            yield (a,cmd)

    def test_jacobi_module(self,a):
        jacobi = eval("la.jacobi")
        res = jacobi(a[0],a[1])
        return (res,"res = la.jacobi(a[0],a[1]);")
