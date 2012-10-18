import cphvbnumpy as np
import util

B = util.Benchmark()
N = B.size[0]
I = B.size[1]

A = np.random.random([N,N])
X = np.random.random([N])
h = np.empty([N], dtype=np.double)
h[:] = 0.001

AD = np.diagonal(A).copy()

A.cphvb     = B.cphvb
X.cphvb     = B.cphvb
h.cphvb     = B.cphvb
AD.cphvb    = B.cphvb

B.start()
for i in xrange(I):
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

