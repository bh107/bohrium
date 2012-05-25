import numpy as np
import numpytest
import cphvbbridge as cp
import unittest

dtype = np.float32

class Bugs(unittest.TestCase):

    def test_leaking(self):

        a = np.ones((128,62768))
        a.cphvb = True
        b = np.ones((128,62768))
        b.cphvb = True

        i = 0

        while True:
            a = b + b + b

            i += 1
            if i > 20:
                break

        print "Out."
        print a
        self.assertTrue(True)

def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(Bugs)
    unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == "__main__":
    unittest.main(verbosity=3)
