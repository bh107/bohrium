"""
Visualization
~~~~~~~~~~~~~

Common functions for visualization.
"""
import bohrium as np
from . import ufuncs, array_create
from bohrium import _bh


def compressed_copy(ary, param):
    a_min = ary.min()
    a_range = ary.max() - ary.min() + 1
    # Normalize `ary` into uint8
    a = (ary - a_min) * 256 / a_range
    assert (a.min() >= 0)
    assert (a.max() < 256)
    a = np.array(a, dtype=np.uint8)
    # Copy `a` using `param`
    a = _bh.mem_copy(a, param=param)
    # un-normalize and convert back to the original dtype of `ary`
    a = array_create.array(a, dtype=ary.dtype)
    return (a * a_range + a_min) / 256.0


def plot_surface(ary, mode, colormap, lowerbound, upperbound, param=None):
    mode = mode.lower()

    ranks = [2, 3]
    modes = ["2d", "3d"]
    types = [np.float32]

    if not (ary.ndim == 2 or ary.ndim == 3):
        raise ValueError("Unsupported array-rank, must be one of %s" % ranks)
    if mode not in modes:
        raise ValueError("Unsupported mode, must be one of %s" % modes)

    if ary.dtype not in types:
        ary = array_create.array(ary, bohrium=True, dtype=np.float32)

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

    args = array_create.array([  # Construct arguments
        np.float32(colormap),
        np.float32(flat),
        np.float32(cube),
        np.float32(lowerbound),
        np.float32(upperbound)
    ],
        bohrium=True,
        dtype=np.float32
    )

    if param is not None:
        ary = compressed_copy(ary, param)

    ufuncs.extmethod("visualizer", ary, args, ary)  # Send to extension
