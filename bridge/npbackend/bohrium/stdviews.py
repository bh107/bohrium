"""
View Constructors
=================
"""

from __future__ import print_function
import itertools
from .bhary import fix_biclass_wrapper


@fix_biclass_wrapper
def cartesian(x, size):
    dist = []
    for i in range(size):
        start = i
        stop = -size + 1 + i
        if stop == 0: stop = None
        dist.append((start, stop))
    stencil = [x[tuple(s)] for s in [map((lambda se: slice(se[0], se[1])), i)
                                     for i in itertools.product(dist,
                                                                repeat=len(x.shape))]]
    return stencil


@fix_biclass_wrapper
def no_border(x, boarder):
    stencil = [x[tuple(s)] for s in [map((lambda se: slice(se[0], se[1])), i)
                                     for i in itertools.product([(boarder, -boarder)],
                                                                repeat=len(x.shape))]]
    return stencil[0]


@fix_biclass_wrapper
def D2P9(x):
    if len(x.shape) != 2:
        raise Exception('Invalid shape for stencil' + str(len(x)))
    return cartesian(x, 3)


@fix_biclass_wrapper
def D3P27(x):
    if len(x.shape) != 3:
        print(len(x))
        raise Exception('Invalid shape for stencil')
    return cartesian(x, 3)


@fix_biclass_wrapper
def D2P8(x):
    result = D2P9(x)
    result.pop(4)
    return result


@fix_biclass_wrapper
def D3P26(x):
    result = D3P27(x)
    result.pop(13)
    return result


@fix_biclass_wrapper
def grid(x, size):  # 2d for now - should be made nd
    if size != 1:
        raise Exception("Only distance 1 supported")

    data = D2P9(x)
    result = []
    for i in range(9):
        if i % 2 != 0:
            result.append(data[i])

    return result


@fix_biclass_wrapper
def diagonals(x, size):
    if size != 1:
        raise Exception("Only distance 1 supported")

    data = D2P9(x)
    result = []
    for i in range(9):
        if i % 2 == 0:
            result.append(data[i])
    result.pop(2)
    return result
