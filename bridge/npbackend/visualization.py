"""
Visualization
~~~~~~~~~~~~~

Common functions for visualization.
"""
import bohrium as np
from . import ufunc, ndarray, array_create

def plot_surface(ary, mode, colormap, lowerbound, upperbound):

    mode = mode.lower()

    ranks = [2, 3]
    modes = ["2d", "3d"]

    if not (ary.ndim == 2 or ary.ndim == 3):
        raise ValueError("Unsupported array-rank, must be one of %s" % ranks)
    if mode not in modes:
        raise ValueError("Unsupported mode, must be one of %s" % modes)
    if not ndarray.check(ary):
        raise ValueError("Input-array must be a Bohrium array")

    if mode == "2d":
        flat = True
        cube = False
    elif mode == "3d":
        if ary.ndim == 2:
            flat = False
            cube = False
        else:
            flat = False
            cube = True
    else:
        raise ValueError("Unsupported mode '%s' " % mode)

    args = array_create.array([                     # Construct arguments
            float(colormap),
            float(flat),
            float(cube),
            float(lowerbound),
            float(upperbound)
        ],
        bohrium=True
    )
    ufunc.extmethod("visualizer", ary, args, ary)   # Send to extension

