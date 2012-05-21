import numpy as np
import numpytest
import cphvbbridge as cp
import unittest

dtype = np.float32

class Bugs(unittest.TestCase):

    def test_leaking(self):

        a = np.ones((1024,1024))
        a.cphvb = True
        b = np.ones((1024,1024))
        b.cphvb = True

        i = 0

        while True:
            a = a + b
            i += 1
            if i > 200:
                break

def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(Traversal)
    unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == "__main__":
    unittest.main(verbosity=3)
