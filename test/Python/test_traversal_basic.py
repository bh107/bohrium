import numpy as np
import numpytest
import cphvbbridge as cp
import unittest

dtype = np.float32

class Traversal(unittest.TestCase):

    def test_3x3x3x3(self):
        res = np.array(np.ones((3,3,3,3))+np.ones((3,3,3,3)))
    
        a = np.array(np.ones((3,3,3,3)),dtype=dtype, cphvb=True)
        b = np.array(np.ones((3,3,3,3)),dtype=dtype, cphvb=True)
        c = a+b

        self.assertTrue(numpytest.array_equal( c, res ))

    def test_3x3x3(self):
        res = np.array(np.ones((3,3,3))+np.ones((3,3,3)))
        a = np.array(np.ones((3,3,3)),dtype=dtype, cphvb=True)
        b = np.array(np.ones((3,3,3)),dtype=dtype, cphvb=True)
        c = a+b

        self.assertTrue(numpytest.array_equal( c, res ))

    def test_3x3(self):
        res = np.array(np.ones((3,3))+np.ones((3,3)))
        a = np.array(np.ones((3,3)),dtype=dtype, cphvb=True)
        b = np.array(np.ones((3,3)),dtype=dtype, cphvb=True)
        c = a+b


        self.assertTrue(numpytest.array_equal( c, res ))

def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(Traversal)
    unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == "__main__":
    unittest.main(verbosity=3)
