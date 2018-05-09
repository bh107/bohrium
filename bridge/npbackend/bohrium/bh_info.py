"""
Bohrium Info
============
"""
import os
from os.path import join
from . import _info

# For convenience, we include functions from other modules that also returns relevant Bohrium information
from .backend_messaging import runtime_info, statistic

# Some cached info
_opencl_is_in_stack = None
_cuda_is_in_stack = None
_proxy_is_in_stack = None


def config_file_path():
    """Return the path to the Bohrium config file in use"""

    if "BH_CONFIG" in os.environ:
        return os.environ["BH_CONFIG"]

    paths = [join(os.path.dirname(os.path.realpath(__file__)), "config.ini"),
             "/usr/local/etc/bohrium/config.ini",
             "/etc/bohrium/config.ini",
             os.path.expanduser(join("~", ".bohrium", "config.ini"))]
    for p in paths:
        if os.path.exists(os.path.realpath(p)):
            return os.path.realpath(p)
    raise RuntimeError("Couldn't find any config file!")


def installed_through_pypi():
    """Return True when Bohrium has been installed through a PyPI package"""
    return os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.ini"))


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


def is_opencl_in_stack():
    """Returns True when the OpenCL backend is in the Bohrium backend"""
    global _opencl_is_in_stack
    if _opencl_is_in_stack is None:
        _opencl_is_in_stack = "OpenCL" in runtime_info()
    return _opencl_is_in_stack


def is_cuda_in_stack():
    """Returns True when the CUDA backend is in the Bohrium backend"""
    global _cuda_is_in_stack
    if _cuda_is_in_stack is None:
        _cuda_is_in_stack = "CUDA" in runtime_info()
    return _cuda_is_in_stack


def is_proxy_in_stack():
    """Returns True when the Proxy component is in the Bohrium backend"""
    global _proxy_is_in_stack
    if _proxy_is_in_stack is None:
        _proxy_is_in_stack = "Proxy" in runtime_info()
    return _proxy_is_in_stack


def pprint():
    """Pretty print Bohrium info"""

    ret = ""
    if not (is_opencl_in_stack() or is_cuda_in_stack()):
        ret += "Note: in order to activate and retrieve GPU info, set the `BH_STACK=opencl` " \
               "or `BH_STACK=cuda` environment variable.\n"

    ret += """----
Bohrium version: %s
Build with NumPy version: %s
Installed through PyPI: %s
Config file: %s
Backend stack:
%s----
""" % (version(), numpy_version(), installed_through_pypi(), config_file_path(), runtime_info())

    return ret
