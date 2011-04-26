import numpy as np
import time
import sys

d = int(sys.argv[1]) #CUDA
N = int(sys.argv[2]) #Size of Model
I = int(sys.argv[3]) #Number of iterations

def monte_carlo_pi(N,I,x,y,t,r):
    sum = 0.0
    for i in range(I):
        if d:
            np.core.multiarray.random(x)
            np.core.multiarray.random(y)
        else:
            x = np.random.rand(N)
            y = np.random.rand(N)
        x *= x
        y *= y    
        x += y
        np.less_equal(x,1.0,t)
        np.add.reduce(t,out=r)
        sum += r[0]*4.0/N
    return sum / I
    
x1 = np.empty(N, dtype=np.float32, dist=d)
y1 = np.empty(N, dtype=np.float32, dist=d)
t1 = np.empty(N, dtype=np.bool, dist=d)
r = np.empty(1, dtype=np.float32, dist=d)

start = time.time()
pi = monte_carlo_pi(N,I,x1,y1,t1,r)
np.core.multiarray.evalflush()
end = time.time()

print d, " ", N, " ", I, " ", end-start
