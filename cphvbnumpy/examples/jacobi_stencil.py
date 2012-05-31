import cphvbnumpy as np

def frezetrap(height, width, dtype=np.float32, cphvb=True):
    grid = np.zeros((height+2,width+2), dtype=dtype, cphvb=cphvb)
    grid[:,0]  = -273.15
    grid[:,-1] = -273.15
    grid[-1,:] = -273.15
    grid[0,:]  =   40.0
    return grid

def solve(grid, epsilon=0.005):
    center = grid[1:-1, 1:-1]
    north  = grid[0:-2, 1:-1]
    east   = grid[1:-1, 2:  ]
    west   = grid[1:-1, 0:-2]
    south  = grid[2:  , 1:-1]

    delta = epsilon + 1
    while delta > epsilon:
        work = 0.2*(center+north+east+west+south)
        delta = np.amax(np.absolute(work-center))
        center[:] = work
    return grid

