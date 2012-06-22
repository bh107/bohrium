import cphvbnumpy as np

def frezetrap(height, width, dtype=np.float32, cphvb=True):
    grid = np.zeros((height+2,width+2), dtype=dtype, cphvb=cphvb)
    grid[:,0]  = -273.15
    grid[:,-1] = -273.15
    grid[-1,:] = -273.15
    grid[0,:]  =   40.0
    return grid

def solve(grid, epsilon=0.005, max_iterations=None):
    center = grid[1:-1, 1:-1]
    north  = grid[0:-2, 1:-1]
    east   = grid[1:-1, 2:  ]
    west   = grid[1:-1, 0:-2]
    south  = grid[2:  , 1:-1]
    delta = epsilon + 1
    iteration = 0
    while delta > epsilon:
        iteration += 1
        work = 0.2*(center+north+east+west+south)
        delta = np.amax(np.absolute(work-center))
        center[:] = work
        if max_iterations != None and max_iterations <= iteration:
            return grid
    return grid

def iterate(grid, iterations):
    center = grid[1:-1, 1:-1]
    north  = grid[0:-2, 1:-1]
    east   = grid[1:-1, 2:  ]
    west   = grid[1:-1, 0:-2]
    south  = grid[2:  , 1:-1]
    for i in xrange(iterations):
        center[:] = 0.2*(center+north+east+west+south)
    return grid

