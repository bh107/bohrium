import numpy as np
import numpytest
import cphvbbridge as cp
import unittest

dtype = np.float32

class Traversal(unittest.TestCase):

    def test_a_bad_case(self):
        a = np.array([[1]]*10,dtype=dtype, cphvb=True)
        b = np.array([[1]]*10,dtype=dtype, cphvb=True)
        c = a+b+a+b

    def test_b_better_case(self):
        a = np.array([[1]*10],dtype=dtype, cphvb=True)
        b = np.array([[1]*10],dtype=dtype, cphvb=True)
        c = a+b+a+b

    def test_c_better_larger_case(self):
        a = np.array([[[1]*10]*10],dtype=dtype, cphvb=True)
        b = np.array([[[1]*10]*10],dtype=dtype, cphvb=True)
        c = a+b+a+b

    def test_oooo(self):
        a = np.ones((20,20))
        a.cphvb = True
        b = np.ones((20,20))
        b.cphvb = True

        c = a[1::2]
        d = b[0::2]
        e = c+d+c+d

def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(Traversal)
    unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == "__main__":
    unittest.main(verbosity=3)
