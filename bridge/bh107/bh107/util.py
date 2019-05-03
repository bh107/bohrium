# -*- coding: utf-8 -*-
"""
========================
Useful Utility Functions
========================
"""

import operator
import functools


def total_size(shape):
    """Returns the total size of the values in `shape`"""
    return 0 if len(shape) == 0 else functools.reduce(operator.mul, shape, 1)


def get_contiguous_strides(shape):
    """Returns a new strides that corresponds to a contiguous traversal of shape"""
    stride = [0] * len(shape)
    s = 1
    for i in reversed(range(len(shape))):
        stride[i] = s
        s *= shape[i]
    return tuple(stride)
