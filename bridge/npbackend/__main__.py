#!/usr/bin/env python

#In this module we implement an "as numpy" hack which makes it possible to
#utilize Bohrium using the command line argument "python -m bohrium.as_numpy"

import sys
import os
import numpy
import bohrium

def get_execfile():
    """execfile() does not exist in Python 3"""
    try:
        return execfile
    except NameError:
        import runpy
        return runpy.run_path


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
    get_execfile()(sys.argv[0])
else:
    print ('ERR: the "-m bohrium" does not support interactive mode')

