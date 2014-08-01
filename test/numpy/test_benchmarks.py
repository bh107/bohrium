import subprocess
import pprint
import pickle

import bohrium as np
from numpytest import numpytest
import bohrium.linalg as la
from bohrium import examples as exp

class test_jacobi(numpytest):#disabled
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.size = 20
    def init(self):
        print "We need to implement numpy.norm() for test_jacobi() to work"
        raise StopIteration()
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

def benchrun(script, size, backend, outputfn):
    """Run the benchmark script and return the result."""

    cmd = [
        'python',
        script,
        '--size='       +size,
        '--backend='    +backend,
        '--outputfn='   +outputfn
    ]
    p = subprocess.Popen(cmd)
    out, err=p.communicate()

    res = None
    with open(outputfn) as fd:
        res = pickle.load(fd)

    return res

class test_gameoflife(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.size = 20

    def init(self):
        yield ({0:np.ones(self.size, bohrium=False)}, "gameoflife")

    def test_gameoflife(self, a):

        backend = "None"
        if 'bohrium.ndarray' in str(type(a[0])):
            backend = "Bohrium"

        res = benchrun('/home/safl/Desktop/bohrium/benchmark/Python/gameoflife.py',
            "20*20*2",
            backend,
            '/tmp/bohrium'
        )
        res['res'] = np.array(res['res'], bohrium=backend!="None")
        return (res['res'], "gameoflife")
"""
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

class test_gameoflife(numpytest):
    def init(self):
        a = {}
        cmd  = "a[0] = exp.gameoflife.randomstate({0},{0});".format(10)
        exec cmd
        yield (a,cmd)

    def test_gameoflife(self,a):
        cmd = "res = exp.gameoflife.play(a[0].copy(), 50);"
        exec cmd
        return (res,cmd)

class test_shallow_water(numpytest):
    def __init__(self):
        numpytest.__init__(self)
        self.config['maxerror'] = 0.00001
        self.size = 20
    def init(self):
        for t in ['np.float32','np.float64']:
            a = {}
            cmd  = "a[0] = exp.shallow_water.model({0},{0},dtype={1});".format(self.size,t)
            exec cmd
            yield (a,cmd)

    def test_shallow_water(self,a):
        cmd = "res = exp.shallow_water.simulate(a[0],10);"
        exec cmd
        return (res,cmd)
"""
