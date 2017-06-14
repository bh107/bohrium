"""
Send and receive a message through the Bohrium component stack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from .target import message as _backend_msg


def statistic_enable_and_reset():
    """Reset and enable the Bohrium statistic"""
    return _backend_msg("statistic_enable_and_reset")


def statistic():
    """Return a YAML string of Bohrium statistic"""
    return _backend_msg("statistic")


def gpu_disable():
    """Disable the GPU backend in the current runtime stack"""
    return _backend_msg("GPU: disable")


def gpu_enable():
    """Enable the GPU backend in the current runtime stack"""
    return _backend_msg("GPU: enable")


def runtime_info():
    """Return a YAML string describing the current Bohrium runtime"""
    return _backend_msg("info")
