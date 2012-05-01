import numpy as np
import numpytest
import cphvbnumpy as cp
import unittest
import util

class Reduction(unittest.TestCase):

    def setUp(self):

        a_0d = [1]     # A "scalar"
        a_1d = [a_0d]*10
        a_2d = [a_1d]*10
        a_3d = [a_2d]*10
        a_4d = [a_3d]*10

        self.a1_0d = np.array(a_0d, dist=True)
        self.a1_1d = np.array(a_1d, dist=True)
        self.v1_1d = self.a1_1d[0:]
        self.a1_2d = np.array(a_2d, dist=True)
        self.a1_3d = np.array(a_3d, dist=True)
        self.a1_4d = np.array(a_4d, dist=True)
        self.v1_4d = self.a1_4d[0:]

    # One-dimensional with only one element
    def test_reduce_1d(self):
        res = 0
        for _ in xrange(0,10):
            res += np.add.reduce( self.a1_1d )*4/3
        res /= 10

    def test_reduce_1dv(self):
        res = np.add.reduce( self.v1_1d )
        cp.unhandle_array(res)

    def test_reduce_2d(self):
        res = np.add.reduce( self.a1_2d )
        cp.unhandle_array(res)

    def test_reduce_3d(self):
        res = np.add.reduce( self.a1_3d )
        cp.unhandle_array(res)

    def test_reduce_4d(self):
        res = np.add.reduce( self.a1_4d )
        cp.unhandle_array(res)

    def test_reduce_4dv(self):
        res = np.add.reduce( self.v1_4d )
        cp.unhandle_array(res)

def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(Reduction)
    unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == "__main__":
    unittest.main(verbosity=3)
