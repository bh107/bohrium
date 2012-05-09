import numpy as np
import cphvbbridge as cp
import unittest

class RandomUfunc(unittest.TestCase):

    def test_array_reduce(self):

        x = np.array([1], cphvb=True)
        r = np.add.reduce(x)

    def test_array_reduce_equal(self):

        x = np.array([1], cphvb=False)
        z = np.less_equal(x, 1.0)
        r = np.add.reduce(z)

    def test_random_reduce(self):

        x = np.random.random([1], cphvb=True)
        r = np.add.reduce(x)

    def test_random_reduce_equal(self):

        x = np.random.random([1], cphvb=True)
        z = np.less_equal(x, 1.0)
        r = np.add.reduce(z)

    def test_random_print(self):

        x = np.random.random([1], cphvb=True)
        print x

def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(RandomUfunc)
    unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == "__main__":
    unittest.main(verbosity=3)
