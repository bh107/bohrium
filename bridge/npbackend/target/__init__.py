"""
Initialization of npbackend target.

The target chosen is controlled by setting one of the following 
environment vars:

    * BHPY_BACKEND
    * BHPY_TARGET
    * NPBE_TARGET

The envvars are searched in the order above, first match wins.

The envvar can be one of the following:

    * bhc
    * numpy
    * numexpr
    * pygpu
    * chapel

If no envvar is set, the default target "bhc" is used.
"""
import os

DEFAULT_TARGET = "bhc"
TARGETS = ["bhc", "numpy", "numexpr", "pygpu", "chapel"]
METHODS = [
    'Base', 'View', 'get_data_pointer', 'set_bhc_data_from_ary',
    'ufunc', 'reduce', 'accumulate', 'extmethod', 'range', 'random123'
]

TARGET = None               # Check for environment override of default target
for env_var in ['BHPY_BACKEND', 'BHPY_TARGET', 'NPBE_TARGET']:
    TARGET = os.getenv(env_var)
    if TARGET:
        break
TARGET = TARGET if TARGET else DEFAULT_TARGET

if TARGET not in TARGETS:   # Validate the target
    msg = "Unsupported npbackend target: %s. Use one of: %s." % (
        TARGET,
        ", ".join(TARGETS)
    )
    raise RuntimeError(msg)

# Do the actual import of methods from the chosen target
exec("from .target_%s import %s" % (TARGET, ", ".join(METHODS)))

