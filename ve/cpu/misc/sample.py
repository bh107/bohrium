import argparse
import pprint
import os

def sample(args):
    if args.be == 'bohrium':
        import bohrium as np
    elif args.be == 'numpy':
        import numpy as np

    a = np.ones(args.shape, bohrium=False)
    b = np.ones(args.shape, bohrium=False)
    c = np.ones(args.shape, bohrium=False)
    a.bohrium=True
    b.bohrium=True
    c.bohrium=True
    np.add(a,b,c)
    a.bohrium=False
    b.bohrium=False
    c.bohrium=False

    return a,b,c
    #a = np.arange(np.prod(args.shape)).reshape(args.shape)
    #b = np.cumsum(a,0)
    #c = np.cumsum(a,1)
    #d = np.add.reduce(a,2)
    #return a, b, c
    #return a, b, c, d

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

    pprint.pprint(sample(args))

if __name__ == "__main__":
    main()
