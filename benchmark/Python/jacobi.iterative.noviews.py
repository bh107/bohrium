import bohrium as np
import util

B = util.Benchmark()
N = B.size[0]
I = B.size[1]

A = np.random.random([N,N],         bohrium=B.bohrium)
X = np.random.random([N],           bohrium=B.bohrium)
h = np.empty([N], dtype=np.double,  bohrium=B.bohrium)
h[:] = 0.001

AD = np.diagonal(A).copy()

AD.bohrium    = B.bohrium

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

