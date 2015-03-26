"""
SOR
--------------

So what does this code example illustrate?

r0 b0 r0 b0 r0 b0
b1 r1 b1 r1 b1 r1
r0 b0 r0 b0 r0 b0
b1 r1 b1 r1 b1 r1
r0 b0 r0 b0 r0 b0
b1 r1 b1 r1 b1 r1
r0 b0 r0 b0 r0 b0
b1 r1 b1 r1 b1 r1

"""
from __future__ import print_function
import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np
import util

def freezetrap(height, width, dtype=np.float32):
    r0   = np.zeros(((height+2)/2,(width+2)/2), dtype=dtype)
    r1   = np.zeros(((height+2)/2,(width+2)/2), dtype=dtype)
    b0   = np.zeros(((height+2)/2,(width+2)/2), dtype=dtype)
    b1   = np.zeros(((height+2)/2,(width+2)/2), dtype=dtype)
    r0[0,:]  =   40.0  # Top
    b0[0,:]  =   40.0  # Top
    r1[-1,:] = -273.15 # Bottom
    b1[-1,:] = -273.15 # Bottom
    r0[:,0]  = -273.15 # Left
    b1[:,0]  = -273.15 # Left
    b0[:,-1] = -273.15 # Right
    r1[:,-1] = -273.15 # Right
    return (r0,r1,b0,b1)

def solve(grid, epsilon=0.005, max_iterations=None):
    r0 = grid[0]
    r1 = grid[1]
    b0 = grid[2]
    b1 = grid[3]
    delta = epsilon + 1
    iteration = 0
    while delta > epsilon:
        iteration += 1
        r1[:-1,:-1] = (r1[:-1,:-1] + b0[:-1,:-1] + b0[1:,:-1] + b1[:-1,:-1] + b1[:-1,1:])*0.2
        r0[1:,1:]   = (r0[1:,1:] + b1[:-1,1:] + b1[1:,1:] + b0[1:,:-1] + b0[1:,1:])*0.2
        b1[:-1,1:]  = (b1[:-1,1:] + r0[:-1,1:] + r0[1:,1:] + r1[:-1,:-1] + r1[:-1,1:])*0.2
        b0[1:,:-1]  = (b0[1:,:-1] + r1[:-1,:-1] + r1[1:,:-1] + r0[1:,:-1] + r0[1:,1:])*0.2
        delta = np.sum(np.absolute(r0[1:,1:]   - b0[1:,:-1])) + \
                np.sum(np.absolute(r1[:-1,:-1] - b1[:-1,1:])) + \
                np.sum(np.absolute(r0[1:,1:]   - b1[:-1,1:])) + \
                np.sum(np.absolute(b0[1:,:-1]  - r1[:-1,:-1]))
        if max_iterations != None and max_iterations <= iteration:
            break
    return (r0,r1,b0,b1)

def iterate(grid, iterations):
    r0 = grid[0]
    r1 = grid[1]
    b0 = grid[2]
    b1 = grid[3]
    for i in xrange(iterations):
        r1[:-1,:-1] = (r1[:-1,:-1] + b0[:-1,:-1] + b0[1:,:-1] + b1[:-1,:-1] + b1[:-1,1:])*0.2
        r0[1:,1:]   = (r0[1:,1:] + b1[:-1,1:] + b1[1:,1:] + b0[1:,:-1] + b0[1:,1:])*0.2
        b1[:-1,1:]  = (b1[:-1,1:] + r0[:-1,1:] + r0[1:,1:] + r1[:-1,:-1] + r1[:-1,1:])*0.2
        b0[1:,:-1]  = (b0[1:,:-1] + r1[:-1,:-1] + r1[1:,:-1] + r0[1:,:-1] + r0[1:,1:])*0.2
    return (r0,r1,b0,b1)

def main():
    B = util.Benchmark()
    H = B.size[0]
    W = B.size[1]
    I = B.size[2]

    ft = freezetrap(H, W, dtype=B.dtype)

    B.start()
    ft = solve(ft,max_iterations=I)
    R = ft[0] + ft[1] + ft[2] + ft[3]
    B.stop()

    B.pprint()
    if B.verbose:
        print(R)
    if B.outputfn:
        B.tofile(B.outputfn, {'res': R})

if __name__ == "__main__":
    main()
