import numpy as np
import numpytest
import random
import unittest

class IdentityElemOp(unittest.TestCase):

    def test_basic(self):

        max_ndim = 4
        for i in range(1,max_ndim+1):
            src = numpytest.random_list(random.sample(range(3, 32),i))
            Ad = np.array(src, dtype=float, cphvb=True)
            Af = np.array(src, dtype=float, cphvb=False)
            Ad[1:] = Ad[:-1]
            Af[1:] = Af[:-1]
            np.add(Ad[:-1], 42, Ad[1:]);
            np.add(Af[:-1], 42, Af[1:]);
            self.assertTrue( numpytest.array_equal(Ad,Af) )

    def test_bool_int(self):

        x = np.array( [False]*10, dtype=bool, cphvb=True) 
        r = np.array( [1]*10, dtype=int, cphvb=True) 
        x += 1

        self.assertTrue( numpytest.array_equal( x, r ) )

    def test_int_bool(self):

        x = np.array( [0]*10, dtype=int, cphvb=True) 
        r = np.array( [True]*10, dtype=bool, cphvb=True) 
        x += True

        self.assertTrue( numpytest.array_equal( x, r ) )

def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(IdentityElemOp)
    unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == "__main__":
    unittest.main()
