import numpy as np
import numpytest
import cphvbbridge as cp
import unittest
import time

dtype = np.float32

class Bugs(unittest.TestCase):

    def test_leaking(self):

        a = np.ones((1024*1024*30))
        a.cphvb = True
        b = np.ones((1024*1024*30))
        b.cphvb = True

        i = 0
        lim = 500
        cp.flush()
        start = time.time()
        while True:
            a += b
            #a = a + b

            i += 1
            if i > lim:
                break

        cp.flush()
        print "Elapsed", time.time()-start
        
        self.assertTrue(True)

def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(Bugs)
    unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == "__main__":
    unittest.main(verbosity=3)
