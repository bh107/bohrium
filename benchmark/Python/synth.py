import bohrium as np
import util

B = util.Benchmark()
N, I = B.size

a = np.ones(N)
b = np.ones(N)
c = np.ones(N)

B.start()
for i in xrange(I):
    t = a+b+c+a+b+c+a+b+c+a+b+c+a+b+c+a+b+c 

B.stop()
B.pprint()

if B.verbose:
    print t
