import argparse
import pprint
import os

def sample(args):
    import bohrium as np
    with_bohrium= (args.be == 'bohrium')

    a = np.ones((3,3),bohrium=with_bohrium)
    b = np.ones((3,3),bohrium=with_bohrium)

    return a*b*b*b*a*b*a

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
