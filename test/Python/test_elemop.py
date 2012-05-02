"""
    This test asserts very basic functionality of elementwise operations.
"""
import numpy as np
import numpytest
import cphvbnumpy as cp
import unittest

class ElementOperators(unittest.TestCase):

    def setUp(self):

        a_0d = [1]     # A "scalar"
        a_1d = [a_0d]*10
        a_2d = [a_1d]*10
        a_3d = [a_2d]*10

        self.a1_0d = np.array(a_0d, cphvb=True)
        self.a2_0d = np.array(a_0d, cphvb=True)
        self.a3_0d = np.array(a_0d, cphvb=True)

        self.a1_1d = np.array(a_1d, cphvb=True)
        self.a2_1d = np.array(a_1d, cphvb=True)
        self.a3_1d = np.array(a_1d, cphvb=True)

        self.a1_1d = np.array(a_1d, cphvb=True)
        self.a2_1d = np.array(a_1d, cphvb=True)
        self.a3_1d = np.array(a_1d, cphvb=True)

        self.a1_2d = np.array(a_2d, cphvb=True)
        self.a2_2d = np.array(a_2d, cphvb=True)
        self.a3_2d = np.array(a_2d, cphvb=True)

        self.a1_3d = np.array(a_3d, cphvb=True)
        self.a2_3d = np.array(a_3d, cphvb=True)
        self.a3_3d = np.array(a_3d, cphvb=True)

        self.v1_3d = self.a1_3d[0:]

        self.ra_0d  = np.array(a_0d)+1
        self.ra_1d  = np.array(a_1d)+1
        self.ra_2d  = np.array(a_2d)+1
        self.ra_3d  = np.array(a_3d)+1

        self.con1 = 1
        self.con2 = 2

    # One-dimensional with only one element
    def test_add_0d_aaa(self):
        self.a1_0d = self.a2_0d + self.a3_0d
        self.assertTrue(numpytest.array_equal(self.a1_0d, self.ra_0d))

    def test_add_0d_aac(self):
        self.a1_0d = self.a2_0d + self.con1
        self.assertTrue(numpytest.array_equal(self.a1_0d, self.ra_0d))

    def test_add_0d_aca(self):
        self.a1_0d = self.con1 + self.a2_0d
        self.assertTrue(numpytest.array_equal(self.a1_0d, self.ra_0d))

    # One-dimensional
    def test_add_1d_aaa(self):
        self.a1_1d = self.a2_1d + self.a3_1d
        self.assertTrue(numpytest.array_equal(self.a1_1d, self.ra_1d))

    def test_add_1d_aac(self):
        self.a1_1d = self.a2_1d + self.con1
        self.assertTrue(numpytest.array_equal(self.a1_1d, self.ra_1d))

    def test_add_1d_aca(self):
        self.a1_1d = self.con1 + self.a2_1d
        self.assertTrue(numpytest.array_equal(self.a1_1d, self.ra_1d))

    # Two-dimensional
    def test_add_2d_aaa(self):
        self.a1_2d = self.a2_2d + self.a3_2d
        self.assertTrue(numpytest.array_equal(self.a1_2d, self.ra_2d))

    def test_add_2d_aac(self):
        self.a1_2d = self.a2_2d + self.con1
        self.assertTrue(numpytest.array_equal(self.a1_2d, self.ra_2d))

    def test_add_2d_aca(self):
        self.a1_2d = self.con1 + self.a2_2d
        self.assertTrue(numpytest.array_equal(self.a1_2d, self.ra_2d))

    # Three-dimensional
    def test_add_3d_aaa(self):
        self.a1_3d = self.a2_3d + self.a3_3d
        self.assertTrue(numpytest.array_equal(self.a1_3d, self.ra_3d))

    def test_add_3d_aac(self):
        self.a1_3d = self.a2_3d + self.con1
        self.assertTrue(numpytest.array_equal(self.a1_3d, self.ra_3d))

    def test_add_3d_aca(self):
        self.a1_3d = self.con1 + self.a2_3d
        self.assertTrue(numpytest.array_equal(self.a1_3d, self.ra_3d))

    # Three-dimensional on views
    def test_add_3dv_aaa(self):
        self.assertTrue(numpytest.array_equal(
            self.v1_3d + self.a2_3d,
            self.ra_3d
        ))

    def test_add_3dv_aac(self):
        res = self.v1_3d + self.con1
        self.assertTrue(numpytest.array_equal(res, self.ra_3d))

    def test_add_3dv_aca(self):
        res = self.con1 + self.v1_3d
        self.assertTrue(numpytest.array_equal(res, self.ra_3d))

def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(ElementOperators)
    unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == "__main__":
    unittest.main()
