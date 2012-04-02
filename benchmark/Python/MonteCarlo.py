import numpy as np
import cphvbnumpy as cnp
import util

B = util.Benchmark()
N = B.size[0]
I = B.size[1]

B.start()
sum=0.0
for i in xrange(I):
    cnp.flush();
    x = np.random.random([N], dtype=B.dtype, cphvb=B.cphvb)
    y = np.random.random([N], dtype=B.dtype, cphvb=B.cphvb)
    np.square(x,x)
    np.square(y,y)
    np.add(x,y,x)
    z = np.less_equal(x, 1.0)
    sum += np.add.reduce(z)*4.0/N
sum /= I
B.stop()
print "Pi: ", sum
B.pprint()

