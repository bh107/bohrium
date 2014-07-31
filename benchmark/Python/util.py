#!/usr/bin/python
import argparse
import pprint
import time
import sys

import bohrium as np

def t_or_f(arg):
    """Helper function to parse "True/true/TrUe/False..." as bools."""

    ua = str(arg).lower()
    if ua == 'true'[:len(ua)]:
       return True
    elif ua == 'false'[:len(ua)]:
       return False
    else:
        return arg

class Benchmark:
    """
    Helper class to aid running Python/NumPy programs with and without npbackend.

    Use it to sample elapsed time using: start()/stop()
    Pretty-prints results using pprint().
    start()/stop() will send flush signals to npbackend, ensuring that only
    the statements in-between start() and stop() are measured.
    """

    def __init__(self):

        self.__elapsed  = 0.0           # The quantity measured
        self.__script   = sys.argv[0]   # The script being run

        # Just for reference... these are the options parsed from cmd-line.
        options = [
            'size',         'dtype',
            'visualize',    'verbose',
            'backend',      'bohrium'
        ]

        # Construct argument parser
        p = argparse.ArgumentParser(description='Benchmark runner for npbackend.')
        p.add_argument('--size',
                       required = True,
                       help     = "Tell the script the size of the data to work on."
        )
        p.add_argument('--dtype',
                       choices  = ["float32, float64"],
                       default  = "float64",
                       help     = "Tell the the script which primitive type to use."
                                  " (default: %(default)s)"
        )
        p.add_argument('--visualize',
                       choices  = [True, False],
                       default  = False,
                       type     = t_or_f,
                       help     = "Enable visualization in script."
                                  "(default: %(default)s)"
        )
        p.add_argument('--verbose',
                       choices  = [True, False],
                       default  = False,
                       type     = t_or_f,
                       help     = "Print out misc information from script."
                                  " (default: %(default)s)"
        )
        p.add_argument('--backend',
                       choices  = ['None', 'NumPy', 'Bohrium'],
                       default  = "None",
                       help     = "Enable npbackend using the specified backend."
                                  " Disable npbackend using None."
                                  " (default: %(default)s)"
        )
        p.add_argument('--bohrium',
                       choices  = [True, False],
                       default  = False,
                       type     = t_or_f,
                       help     = "Same as --backend=bohrium which means:"
                                  " enable npbackend using bohrium."
                                  " (default: %(default)s)"
        )
        args = p.parse_args()   # Parse the arguments

        #
        # Conveniently expose options to the user
        #
        self.size       = [int(i) for i in args.size.split("*")]
        self.dtype      = eval("np.%s" % args.dtype)

        # Unify the options: 'backend' and 'bohrium'
        if args.bohrium or args.backend == 'bohrium':
            self.backend    = "bohrium"
            self.bohrium    = True
        else:
            self.backend = args.backend
            self.bohrium = args.bohrium

        self.visualize  = args.visualize
        self.verbose    = args.verbose

        #
        # Also make them available via the parser and arg objects
        self.p      = p
        self.args   = args

    def start(self):
        np.flush()
        self.__elapsed = time.time()

    def stop(self):
        np.flush()
        self.__elapsed = time.time() - self.__elapsed

    def pprint(self):
        print "%s - backend: %s, bohrium: %s, size: %s, elapsed-time: %f" % (
                self.__script,
                self.backend,
                self.bohrium,
                '*'.join([str(s) for s in self.size]),
                self.__elapsed
        )

def main():
    B = Benchmark()
    B.start()
    B.stop()
    if B.visualize:
        pprint.pprint(B.args)
    B.pprint()

if __name__ == "__main__":
    main()
