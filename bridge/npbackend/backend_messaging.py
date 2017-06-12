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
