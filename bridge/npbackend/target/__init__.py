"""
Initialization of npbackend target.

The target chosen is controlled by setting one of the following
environment vars:

    * BHPY_BACKEND
    * BHPY_TARGET
    * NPBE_TARGET

The envvars are searched in the order above, first match wins.

The envvar can refer to any of the target_*.py files in this directory.
The default target is bhc, which is the Bohrium-C backend.
"""

import os

DEFAULT_TARGET = "bhc"

def _get_target():
    """Returns the target backend module to use"""

    files = os.listdir(os.path.dirname(os.path.abspath(__file__)))
    targets = []
    for f in files:
        if f.startswith("target_") and f.endswith(".py"):
            targets.append(f)

    target = None               # Check for environment override of default target
    for env_var in ['BHPY_BACKEND', 'BHPY_TARGET', 'NPBE_TARGET']:
        target = os.getenv(env_var)
        if target:
            break
    target = target if target else DEFAULT_TARGET
    target = "target_%s.py"%target

    if target not in targets:   # Validate the target
        msg  = "Unsupported npbackend target: '%s'. "%target[len("target_"):-len(".py")]
        msg += "Use one of: "
        for t in targets:
            msg += "'%s', "%t[len("target_"):-len(".py")]
        msg = "%s."%msg[:-2]
        raise RuntimeError(msg)

    return target

_target = _get_target()

# First we import the basic virtual interface
from .interface import *

# Then we import the target backend implementation
exec("from .%s import *"%_target[:-len(".py")])

#Finally, we expose the chosen target name
TARGET = _target[len("target_"):-len(".py")]

