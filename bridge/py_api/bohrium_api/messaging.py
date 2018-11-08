"""
Send and receive pre-defined messages through the Bohrium component stack
=========================================================================
"""

from ._bh_api import message as msg


def statistic_enable_and_reset():
    """Reset and enable the Bohrium statistic"""
    return msg("statistic_enable_and_reset")


def statistic():
    """Return a YAML string of Bohrium statistic"""
    return msg("statistic")


def gpu_disable():
    """Disable the GPU backend in the current runtime stack"""
    return msg("GPU: disable")


def gpu_enable():
    """Enable the GPU backend in the current runtime stack"""
    return msg("GPU: enable")


def runtime_info():
    """Return a YAML string describing the current Bohrium runtime"""
    return msg("info")


def cuda_use_current_context():
    """Tell the CUDA backend to use the current CUDA context (useful for PyCUDA interop)"""
    return msg("CUDA: use current context")
