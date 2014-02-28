import argparse
import pprint
import os

def sample(args):
    if args.be == 'bohrium':
        import bohrium as np
    elif args.be == 'numpy':
        import numpy as np

    a = np.ones(27)
    return a+a

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
