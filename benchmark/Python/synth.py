import bohrium as np
import util

B = util.Benchmark()
N, I = B.size

B.start()
a = np.ones(N)
b = np.ones(N)
c = np.ones(N)

for _ in xrange(I):
    r = a+b+c

B.stop()
B.pprint()

if B.verbose:
    print r
