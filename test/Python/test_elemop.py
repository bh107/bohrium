"""
    This test asserts very basic functionality of elementwise operations.
"""
import numpy as np
import numpytest
import cphvbnumpy as cp
import unittest

class ElementOperators(unittest.TestCase):

    def setUp(self):
        self.arr1 = np.array([1, 1, 1], dist=True)
        self.arr2 = np.array([1, 1, 1], dist=True)
        self.arr3 = np.array([1, 1, 1], dist=True)

        self.res  = np.array([2, 2, 2])
        self.con1 = 1
        self.con2 = 2

    def test_add_aaa(self):
        self.arr1 = self.arr2 + self.arr3
        self.assertTrue(numpytest.array_equal(self.arr1, self.res))

    def test_add_aac(self):
        self.arr1 = self.arr2 + self.con1
        self.assertTrue(numpytest.array_equal(self.arr1, self.res))

    def test_add_aca(self):
        self.arr1 = self.con1 + self.arr2
        self.assertTrue(numpytest.array_equal(self.arr1, self.res))

def run():
    pass

if __name__ == "__main__":
    unittest.main()

