"""
Open Source Computer Vision (OpenCV)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import bohrium as np
from sys import stderr
from . import ufuncs


def __threshold(name, a, thresh, maxval):
    b = np.array([thresh, maxval], dtype=a.dtype)
    c = np.empty_like(a)
    ufuncs.extmethod(name, c, a, b)
    return c


def threshold(a, thresh=127, maxval=255):
    return __threshold("opencv_threshold_b", a, thresh, maxval)


def threshold_b(a, thresh=127, maxval=255):
    return __threshold("opencv_threshold_b", a, thresh, maxval)


def threshold_bi(a, thresh=127, maxval=255):
    return __threshold("opencv_threshold_bi", a, thresh, maxval)


def threshold_t(a, thresh=127, maxval=255):
    return __threshold("opencv_threshold_t", a, thresh, maxval)


def threshold_tz(a, thresh=127, maxval=255):
    return __threshold("opencv_threshold_tz", a, thresh, maxval)


def threshold_tzi(a, thresh=127, maxval=255):
    return __threshold("opencv_threshold_tzi", a, thresh, maxval)


def connected_components(a, connectivity=8):
    b = np.array([connectivity], dtype=a.dtype)
    c = np.zeros_like(a)
    ufuncs.extmethod("opencv_connected_components", c, a, b)
    return c


def erode(a, b=None, c=None):
    if b is None:
        b = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=a.dtype)

    if c is None:
        c = np.empty_like(a)

    ufuncs.extmethod("opencv_erode", c, a, b)
    return c


def dilate(a, b=None, c=None):
    if b is None:
        b = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=a.dtype)

    if c is None:
        c = np.empty_like(a)

    ufuncs.extmethod("opencv_dilate", c, a, b)
    return c
