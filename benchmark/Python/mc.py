import cphvbnumpy as np
import util

B = util.Benchmark()
I = B.size.pop()
N = np.multiply.reduce(B.size)

B.start()
sum=0.0
for i in xrange(I):

    x = np.random.random(B.size, dtype=B.dtype, cphvb=B.cphvb)
    y = np.random.random(B.size, dtype=B.dtype, cphvb=B.cphvb)

    np.square(x,x)
    np.square(y,y)
    np.add(x,y,x)
    z = np.less_equal(x, 1.0)
    while z.ndim > 1:
        z = np.add.reduce(z)
    sum += np.add.reduce(z)*4.0/N

sum /= I
B.stop()
B.pprint()

