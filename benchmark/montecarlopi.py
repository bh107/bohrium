import numpy as np
import time

d=True

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
#        t = x <= 1.0
        np.less_equal(x,1.0,t)
        np.add.reduce(t,out=r)
        sum += r[0]*4.0/N
    return sum / I
    
N = 16776960
I = 200
x1 = np.empty(N, dtype=np.float32, dist=d)
y1 = np.empty(N, dtype=np.float32, dist=d)
t1 = np.empty(N, dtype=np.bool, dist=d)
r = np.empty(1, dtype=np.float32, dist=d)

start = time.time()
pi = monte_carlo_pi(N,I,x1,y1,t1,r)
end = time.time()
print pi
print end-start
