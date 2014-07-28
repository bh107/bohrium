import bohrium as np
import util

B = util.Benchmark()
S = B.size[0]
I = B.size[1]
m = np.random.random(S**2, dtype=B.dtype, bohrium=False).reshape(S,S)
m = np.array(m, bohrium=B.bohrium)


#NB: for now we simply do matmuls

B.start()
for i in xrange(I):
    m = np.dot(m,m)
B.stop()
B.pprint()
