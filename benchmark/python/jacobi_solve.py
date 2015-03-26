from __future__ import print_function
import util
if util.Benchmark().bohrium:
    import bohrium as np
else:
    import numpy as np

def freezetrap(height, width, dtype=np.float32):
    grid        = np.zeros((height+2,width+2), dtype=dtype)
    grid[:,0]   = dtype(-273.15)
    grid[:,-1]  = dtype(-273.15)
    grid[-1,:]  = dtype(-273.15)
    grid[0,:]   = dtype(40.0)
    return grid

def solve(grid, epsilon=0.005, max_iterations=None, visualize=False):
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
        delta = np.sum(np.absolute(work-center))
        center[:] = work
        if max_iterations != None and max_iterations <= iteration:
            break
        if visualize:
            np.visualize(grid, "2d", 0, 0.0, 5.5)
    return grid

def main():
    B = util.Benchmark()
    H = B.size[0]
    W = B.size[1]
    I = B.size[2]

    if B.inputfn:
        ft = B.load_array()
    else:
        ft = freezetrap(H, W, dtype=B.dtype)

    if B.dumpinput:
        B.dump_arrays("jacobi_solve", {'input': ft})

    B.start()
    ft = solve(ft, max_iterations=I, visualize=B.visualize)
    B.stop()
    if B.verbose:
        print(ft)
    B.pprint()
    if B.outputfn:
        B.tofile(B.outputfn, {'res': ft})

if __name__ == "__main__":
    main()
