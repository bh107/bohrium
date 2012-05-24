import numpy as np
import numpytest
import cphvbbridge as cp
import unittest

dtype = np.float32

class Bundler(unittest.TestCase):

    def test_todo(self):
        pass

def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(Bundler)
    unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == "__main__":
    unittest.main(verbosity=3)
