import argparse
import os

def sample(args):
    if args.be == 'bohrium':
        import bohrium as np
    elif args.be == 'numpy':
        import numpy as np

    grid = np.zeros((args.shape[0]+2,args.shape[0]+2))
    grid[:,0]  = -273.15
    grid[:,-1] = -273.15
    grid[-1,:] = -273.15
    grid[0,:]  = 40.0
    """
    center = grid[1:-1, 1:-1]
    north  = grid[0:-2, 1:-1]
    east   = grid[1:-1, 2:  ]
    west   = grid[1:-1, 0:-2]
    south  = grid[2:  , 1:-1]
    for i in xrange(int(args.iterations)):
        center[:] = 0.2*(center+north+east+west+south)
    """
    return grid

def main():
    p = argparse.ArgumentParser('Run a dummy program')
    p.add_argument(
        'shape', metavar='N', type=int,
        nargs='+', help="Shape of the input."
    )
    p.add_argument(
        'iterations', metavar='I', type=int, help="Number of iterations to run."
    )
    p.add_argument(
        '--be', choices=['bohrium', 'numpy'], default='bohrium',
        help="The backend to use"
    )
    args = p.parse_args()

    print sample(args)

if __name__ == "__main__":
    main()
