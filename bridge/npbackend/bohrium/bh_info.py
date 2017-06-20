"""
Bohrium Info
============
"""

from . import _info

# For convenience, we include functions from other modules that also returns relevant Bohrium information
from .backend_messaging import runtime_info, statistic


def version():
    """Return the version of Bohrium"""
    return _info.__version__


def numpy_version():
    """Return the version of NumPy Bohrium was build with."""
    return _info.version_numpy


def info():
    """Return a dict with all info"""
    return {
        "version": version(),
        "numpy_version": numpy_version(),
        "runtime_info": runtime_info(),
        "statistics": statistic(),
    }
