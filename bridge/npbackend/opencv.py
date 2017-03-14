"""
Open Source Computer Vision (OpenCV)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import bohrium as np
from sys import stderr
from . import ufuncs

def erode(a, b, c=None):
    if c is None:
        c = np.empty_like(a)

    ufuncs.extmethod("opencv_erode", c, a, b)
    return c


def dilate(a, b, c=None):
    if c is None:
        c = np.empty_like(a)

    ufuncs.extmethod("opencv_dilate", c, a, b)
    return c
