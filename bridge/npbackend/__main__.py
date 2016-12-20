#!/usr/bin/env python

#In this module we implement an "as numpy" hack which makes it possible to
#utilize Bohrium using the command line argument "python -m bohrium.as_numpy"

import sys
import os
import numpy
import bohrium

def execfile_wrapper(path):
    """execfile() does not exist in Python 3"""
    try:
        execfile(path)
    except NameError:
        import runpy
        return runpy.run_path(path, run_name="__main__")


# numpy becomes bohrium
sys.modules['numpy_force'] = numpy
sys.modules['numpy'] = bohrium

# Set the module search path to the dir of the script
sys.argv.pop(0)
if len(sys.argv) > 0:
    sys.path[0] = os.path.dirname(os.path.abspath(sys.argv[0]))
else:
    sys.path[0] = ""

# Let's run the script
if len(sys.argv) > 0:
    execfile_wrapper(sys.argv[0])
else:
    print ('ERR: the "-m bohrium" does not support interactive mode')

