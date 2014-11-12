#!/usr/bin/env python

#In this module we implement an "as numpy" hack which makes it possible to
#utilize Bohrium using the command line argument "python -m bohrium.as_numpy"

from __future__ import print_function
import sys
import os
import numpy
import bohrium
sys.modules['numpy_force'] = numpy
sys.modules['numpy'] = bohrium

sys.argv.pop(0)
if len(sys.argv) > 0:
    sys.path[0] = os.path.dirname(os.path.abspath(sys.argv[0]))
else:
    sys.path[0] = ""
if len(sys.argv) > 0:
    execfile(sys.argv[0])
else:
    print('ERR: the "-m bohrium.as_numpy" does not support interactive mode');

