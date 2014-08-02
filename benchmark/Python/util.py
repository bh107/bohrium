#!/usr/bin/python
import argparse
import pprint
import pickle
import time
import sys
import re

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
            'backend',      'bohrium',
            'inputfn',      'outputfn'
        ]

        # Construct argument parser
        p = argparse.ArgumentParser(description='Benchmark runner for npbackend.')

        # We can only have required options when the module is run from
        # command-line. When either directly or indirectly imported
        # we cant.
        owns_main = __name__ == "__main__"
        
        #g1 = p.add_mutually_exclusive_group(required = owns_main)
        p.add_argument('--size',
                       help = "Tell the script the size of the data to work on."
        )
        p.add_argument('--inputfn',
                       help = "Input file to use as data."
        )

        p.add_argument('--dtype',
                       choices  = ["float32", "float64"],
                       default  = "float64",
                       help     = "Tell the the script which primitive type to use."
                                  " (default: %(default)s)"
        )
        p.add_argument('--outputfn',
                       help     = "Output file to store results in."
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

        g2 = p.add_mutually_exclusive_group()
        g2.add_argument('--backend',
                       choices  = ['None', 'NumPy', 'Bohrium'],
                       default  = "None",
                       help     = "Enable npbackend using the specified backend."
                                  " Disable npbackend using None."
                                  " (default: %(default)s)"
        )
        g2.add_argument('--bohrium',
                       choices  = [True, False],
                       default  = False,
                       type     = t_or_f,
                       help     = "Same as --backend=bohrium which means:"
                                  " enable npbackend using bohrium."
                                  " (default: %(default)s)"
        )

        args, unknown = p.parse_known_args()   # Parse the arguments

        #
        # Conveniently expose options to the user
        #
        self.size       = [int(i) for i in args.size.split("*")] if args.size else []
        self.dtype      = eval("np.%s" % args.dtype)

        # Unify the options: 'backend' and 'bohrium'
        if args.bohrium or args.backend.lower() == 'bohrium':
            self.backend    = "bohrium"
            self.bohrium    = True
        else:
            self.backend = args.backend
            self.bohrium = args.bohrium

        self.visualize  = args.visualize
        self.verbose    = args.verbose
        self.inputfn    = args.inputfn
        self.outputfn   = args.outputfn

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

    def tofile(self, arrays, filename=None):
        """
        Writes a dict of arrays such as:

        arrays = {'lbl1': array1, 'lbl1': array2}

        To file, if no filename is given, self.outputfn is used.
        Arrays are stored using pickle.
        """
        outputfn = self.outputfn    # Determine filename
        if filename:
            outputfn = filename
                                    # bh-arrays can't be pickled
        for k in arrays:            # so we copy them first
            if 'bohrium' in str(type(arrays[k])):
                arrays[k] = arrays[k].copy2numpy()
        
        with open(outputfn, 'wb') as fd:
            pickle.dump(arrays, fd)

    def dump_arrays(self, arrays, prefix):
        """
        Dumps a dict of arrays organized such as:
            
        arrays = {'lbl1': array1, 'lbl2': array2}

        Into a file using the following naming convention:
        "prefix_lbl1-DTYPE-SHAPE_lbl2-DTYPE-SHAPE"

        The arrays are stored using pickle.
        """
        names = []
        for k in arrays:
            names.append("%s-%s-%s" % (
                k,
                arrays[k].dtype,
                '*'.join([str(x) for x in (arrays[k].shape)]
            )))
        filename = "%s_%s.pkl" % (prefix, '_'.join(names))
        self.tofile(arrays, filename)

    def load_arrays(self, filename=None):

        if not filename:
            filename = self.inputfn
        
        content = None
        with open(self.inputfn) as fd:
            content = pickle.load(fd)
        return content

    def import_array(self, filename, label='input'):

        array = np.array(
            self.load_arrays(filename)[label],
            bohrium = self.bohrium
        )

        return array

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
