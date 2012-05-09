import numpy as np
import cphvbbridge
import util

B = util.Benchmark()
N = B.size[0]
iterations = B.size[1]

A = np.random.random([N,N], cphvb=B.cphvb)
X = np.random.random([N],   cphvb=B.cphvb)
h = np.empty([N], dtype=np.double, dist=B.cphvb)
h[:] = 0.001
AD = np.diagonal(A).copy()
if B.cphvb:
    cphvbbridge.handle_array(AD)

B.start()
for i in xrange(iterations):
    t1 = A * h
    t1 = np.add.reduce(t1)
    t1 -= X
    t1 /= AD
    h_new =  h + t1
    t2 = h_new - h
    t1 = np.absolute(t2 / h)
    h[:] = h_new
B.stop()
B.pprint()

