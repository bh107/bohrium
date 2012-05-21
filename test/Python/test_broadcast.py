import numpy as np
import numpytest
import cphvbbridge as cp
import unittest

dtype = np.float32

class TraverseBroadcast(unittest.TestCase):

    def test_a_bad_case(self):
        tb3 = np.ones((3,3,3))
        tb2 = np.ones((3,3))
        
        a = np.array(tb3,dtype=dtype, cphvb=True)
        b = np.array(tb2,dtype=dtype, cphvb=True)
        c = a+b

        print c

def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(TraverseBroadcast)
    unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == "__main__":
    unittest.main(verbosity=3)
