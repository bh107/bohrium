#!/usr/bin/env python

# In this module we implement an "as numpy" hack which makes it possible to
# utilize Bohrium using the command line argument "python -m bohrium.as_numpy"

import sys
import os
import argparse
import bohrium
import bohrium_api
from . import version


@bohrium.replace_numpy
def execfile_wrapper(path):
    """execfile() does not exist in Python 3"""

    # We need this ugly code in order to avoid wrapping the script execution in a try/except construct
    try:
        execfile
    except NameError:
        import runpy
        return runpy.run_path(path, init_globals={}, run_name="__main__")
    return execfile(path, {"__name__": "__main__", "__file__": path})


if len(sys.argv) <= 1:
    sys.stderr.write(
        'ERR: the "-m bohrium" does not support interactive mode. Use `-m bohrium --info` for Bohrium info\n')
    sys.exit(-1)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '--info',
    action="store_true",
    default=False,
    help='Print Runtime Info'
)
(args, argv) = parser.parse_known_args()

# If there are more arguments than parsed, we are running a regular script
# Else we are running one of the build-in Bohrium scripts
if len(argv) > 0:
    # Set the module search path to the dir of the script
    sys.argv.pop(0)
    if len(sys.argv) > 0:
        sys.path[0] = os.path.dirname(os.path.abspath(sys.argv[0]))
    else:
        sys.path[0] = ""
    execfile_wrapper(sys.argv[0])
else:
    if args.info:
        print("----\nBohrium version: %s" % version.__version__)
        print(bohrium_api.stack_info.pprint())

        cmd = "import bohrium as bh; import numpy as np; assert((bh.arange(10) == np.arange(10)).all())"
        sys.stdout.write('Sanity Check: "%s"' % cmd)
        try:
            exec (cmd)
            sys.stdout.write(' - success!\n')
        except AssertionError as e:
            sys.stdout.write('\n')
            sys.stderr.write("ERROR - the sanity checked failed!\n")
