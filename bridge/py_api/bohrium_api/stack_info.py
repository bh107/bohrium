"""
Runtime Stack Info
==================
"""
import os
from os.path import join
from . import messaging
from .version import __version__, __version_info__

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
             "/usr/etc/bohrium/config.ini",
             "/etc/bohrium/config.ini",
             os.path.expanduser(join("~", ".bohrium", "config.ini"))]
    for p in paths:
        if os.path.exists(os.path.realpath(p)):
            return os.path.realpath(p)
    raise RuntimeError("Couldn't find any config file!")


def installed_through_pypi():
    """Return True when Bohrium has been installed through a PyPI package"""
    return os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.ini"))


def header_dir():
    """Return the path to the C header directory"""
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "include")


def info():
    """Return a dict with all info"""
    return {
        "config_path": config_file_path(),
        "version": __version__,
        "version_info": __version_info__,
        "runtime_info": messaging.runtime_info(),
        "statistics": messaging.statistic(),
        "header_dir": header_dir(),
    }


def is_opencl_in_stack():
    """Returns True when the OpenCL backend is in the Bohrium backend"""
    global _opencl_is_in_stack
    if _opencl_is_in_stack is None:
        _opencl_is_in_stack = "OpenCL" in messaging.runtime_info()
    return _opencl_is_in_stack


def is_cuda_in_stack():
    """Returns True when the CUDA backend is in the Bohrium backend"""
    global _cuda_is_in_stack
    if _cuda_is_in_stack is None:
        _cuda_is_in_stack = "CUDA" in messaging.runtime_info()
    return _cuda_is_in_stack


def is_proxy_in_stack():
    """Returns True when the Proxy component is in the Bohrium backend"""
    global _proxy_is_in_stack
    if _proxy_is_in_stack is None:
        _proxy_is_in_stack = "Proxy" in messaging.runtime_info()
    return _proxy_is_in_stack


def pprint():
    """Pretty print Bohrium info"""

    ret = """----
Bohrium API version: %s
Installed through PyPI: %s
Config file: %s
Header dir: %s
Backend stack:
%s----
""" % (__version__, installed_through_pypi(), config_file_path(), header_dir(), messaging.runtime_info())
    if not (is_opencl_in_stack() or is_cuda_in_stack()):
        ret += "Note: in order to activate and retrieve GPU info, set the `BH_STACK=opencl` " \
               "or `BH_STACK=cuda` environment variable.\n"
    return ret
