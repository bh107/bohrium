"""
Visualization
~~~~~~~~~~~~~

Common functions for visualization.
"""
import bohrium as np
from . import ufuncs, array_create
from bohrium import _bh


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

    # When param is used, we normalize `ary` to uint8 and use `mem_copy()` to compress `ary`
    if param is not None:
        ary_min = ary.min()
        ary_range = ary.max() - ary.min()
        ary = (ary-ary_min) * 255 / ary_range
        assert(ary.min() >= 0)
        assert(ary.max() < 256)
        ary = np.array(ary, dtype=np.uint8)
        ary = _bh.mem_copy(ary, param=param)
        ary = array_create.array(ary, dtype=np.float32)
        ary = (ary * ary_range + ary_min) / 255.0

    ufuncs.extmethod("visualizer", ary, args, ary)  # Send to extension
