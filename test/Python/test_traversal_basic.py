import numpy as np
import numpytest
import cphvbbridge as cp
import unittest

dtype = np.float32

class Traversal(unittest.TestCase):

    def test_3x3x3x3(self):
        a = np.array(np.ones((3,3,3,3)),dtype=dtype, cphvb=True)
        b = np.array(np.ones((3,3,3,3)),dtype=dtype, cphvb=True)
        c = a+b
        print c

    def test_3x3x3(self):
        a = np.array(np.ones((3,3,3)),dtype=dtype, cphvb=True)
        b = np.array(np.ones((3,3,3)),dtype=dtype, cphvb=True)
        c = a+b
        print c

    def test_3x3(self):
        a = np.array(np.ones((3,3)),dtype=dtype, cphvb=True)
        b = np.array(np.ones((3,3)),dtype=dtype, cphvb=True)
        c = a+b
        print c

def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(Traversal)
    unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == "__main__":
    unittest.main(verbosity=3)
