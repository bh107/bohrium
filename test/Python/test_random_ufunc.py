import numpy as np
import cphvbnumpy as cp
import unittest

class RandomUfunc(unittest.TestCase):

    def test_random_reduce(self):

        a = 0
        x = np.random.random([1], cphvb=True)
        z = np.less_equal(x, 1.0)
        print np.add.reduce(z)

    def test_random_print(self):

        x = np.random.random([1], cphvb=True)
        print x

def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(RandomUfunc)
    unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == "__main__":
    unittest.main(verbosity=3)
