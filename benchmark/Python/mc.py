import bohrium as np
import util

B = util.Benchmark()
N, I = B.size

B.start()
acc=0.0
for i in xrange(I):
    x = np.random.random(N, dtype=B.dtype, bohrium=B.bohrium)
    y = np.random.random(N, dtype=B.dtype, bohrium=B.bohrium)

    z = np.sqrt(x*x+y*y)<=1.0
    acc += np.sum(z)*4.0/N

acc /= I
B.stop()
B.pprint()


print acc
