import numpy as np
import numpytest
import cphvbbridge as cp
import unittest

dtype = np.float32

class Broadcast(unittest.TestCase):

    def test_a_bad_case(self):
        res = np.ones((3,3,3))+np.ones((3,3,3))

        tb3 = np.ones((3,3,3))
        tb2 = np.ones((3,3))
        
        a = np.array(tb3,dtype=dtype, cphvb=True)
        b = np.array(tb2,dtype=dtype, cphvb=True)
        c = a+b

        self.assertTrue(numpytest.array_equal( c, res ))

def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(Broadcast)
    unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == "__main__":
    unittest.main(verbosity=3)
