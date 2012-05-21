import numpy as np
import numpytest
import cphvbbridge as cp
import unittest

dtype = np.float32

class Traversal(unittest.TestCase):

    def test_a_bad_case(self):
        ones = np.ones((10,1))
        res = ones + ones + ones + ones
        a = np.array(ones,dtype=dtype, cphvb=True)
        b = np.array(ones,dtype=dtype, cphvb=True)
        c = a+b+a+b

        self.assertTrue(numpytest.array_equal( c, res ))

    def test_b_better_case(self):
        ones = np.ones((1,10))
        res = ones+ones+ones+ones

        a = np.array(ones,dtype=dtype, cphvb=True)
        b = np.array(ones,dtype=dtype, cphvb=True)
        c = a+b+a+b

        self.assertTrue(numpytest.array_equal( c, res ))

    def test_stride(self):
        ones = np.ones((20,10))
        res = ones+ones+ones+ones

        a = np.ones((20,20))
        a.cphvb = True
        b = np.ones((20,20))
        b.cphvb = True

        c = a[1::2]
        d = b[0::2]
        e = c+d+c+d

        self.assertTrue(numpytest.array_equal( e, res ))

def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(Traversal)
    unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == "__main__":
    unittest.main(verbosity=3)
