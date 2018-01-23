# Python version of the Heat Equation using Bohrium
import numpy as np

# np.set_printoptions(linewidth=200)

def heat2d(height, width, epsilon=42):
    grid = np.zeros((height+2, width+2), dtype=np.float64)
    grid[:,0]  = -273.15
    grid[:,-1] = -273.15
    grid[-1,:] = -273.15
    grid[0,:]  = 40.0

    center = grid[1:-1,1:-1]
    north  = grid[:-2,1:-1]
    east   = grid[1:-1,2:]
    west   = grid[1:-1,:-2]
    south  = grid[2:,1:-1]

    delta = epsilon+1
    while delta > epsilon:
        tmp = 0.2 * (center + north + south + east + west)
        delta = np.sum(np.abs(tmp - center))
        center[:] = tmp

    return center

import sys
size = int(sys.argv[1])
result = heat2d(size, size)
try:
    np.flush()
except:
    pass
