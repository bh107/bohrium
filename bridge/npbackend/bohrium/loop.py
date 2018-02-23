"""
Bohrium Loop
============
"""

import sys
import numpy_force as numpy
from . import _bh
from . import bhary
from . import array_create


def do_while(func, niters, *args, **kwargs):
    """Repeatedly calls the `func` with the `*args` and `**kwargs` as argument.

    The `func` is called while `func` returns True or None and the maximum number
    of iterations, `niters`, hasn't been reached.

    Parameters
    ----------
    func : function
        The function to run in each iterations. `func` can take any argument and may return
        a boolean `bharray` with one element.
    niters: int or None
        Maximum number of iterations in the loop (number of times `func` is called). If None, there is no maximum.
    *args, **kwargs : list and dict
        The arguments to `func`

    Notes
    -----
    `func` can only use operations supported natively in Bohrium.

    Examples
    --------
    >>> def loop_body(a):
    ...     a += 1
    >>> a = bh.zeros(4)
    >>> bh.do_while(loop_body, 5, a)
    >>> a
    array([5, 5, 5, 5])

    >>> def loop_body(a):
    ...     a += 1
    ...     return bh.sum(a) < 10
    >>> a = bh.zeros(4)
    >>> bh.do_while(loop_body, None, a)
    >>> a
    array([3, 3, 3, 3])
    """

    _bh.flush()
    flush_count = _bh.flush_count()
    cond = func(*args, **kwargs)
    if flush_count != _bh.flush_count():
        raise TypeError("Invalid `func`: the looped function contains operations not support "
                        "by Bohrium, contain branches, or is simply too big!")
    if niters is None:
        niters = sys.maxsize-1
    if cond is None:
        _bh.flush_and_repeat(niters, None)
    else:
        if not bhary.check(cond):
            raise TypeError("Invalid `func`: `func` may only return Bohrium arrays or nothing at all")
        if cond.dtype.type is not numpy.bool_:
            raise TypeError("Invalid `func`: `func` returned array of wrong type `%s`. "
                            "It must be of type `bool`." % cond.dtype)
        if len(cond.shape) != 0 and len(cond) > 1:
            raise TypeError("Invalid `func`: `func` returned array of shape `%s`. "
                            "It must be a scalar or an array with one element." % cond.shape)
        if not bhary.is_base(cond):
            raise TypeError("Invalid `func`: `func` returns an array view. It must return a base array.")

        _bh.sync(cond)
        _bh.flush_and_repeat(niters, cond)

def for_loop(loop_body, niters, *args, **kwargs):
    """Calls the `loop_body` with the `*args` and `**kwargs` as argument.

    The `loop_body` is called `niters` times.

    Parameters
    ----------
    loop_body : function
        The function to run in each iterations. `func` can take any arguments.
    niters: int
        Number of iterations in the loop (number of times `loop_body` is called).
    *args, **kwargs : list and dict
        The arguments to `func`

    Notes
    -----
    `func` can only use operations supported natively in Bohrium.

    Examples
    --------
    """

    # The number of iterations must be positive
    if niters < 1: return

    # Clear the cache
    _bh.flush()

    flush_count = _bh.flush_count()
    loop_body(*args, **kwargs)
    if flush_count != _bh.flush_count():
        raise TypeError("Invalid `func`: the looped function contains operations not support "
                        "by Bohrium, contain branches, or is simply too big!")

    _bh.flush_and_repeat(niters, None)

def slide_view(a, dim_stride_tuples):
    """Creates a dynamic view within a loop, that updates the given dimensions by the given strides at the end of each iteration.

    Parameters
    ----------
    a : array view
        A view into an array
    dim_stride_tuples: (int, int)[]
        A list of (dimension, stride) pairs. For each of these pairs, the dimension is updated by the stride in each iteration of a loop.

    Notes
    -----
    No boundary checks are performed. If the view overflows the array, the behaviour is undefined.
    All dyn_views must be at the top of the loop body.
    All views are changed at the end of an iteration and cannot be performed in the middle of a loop body.

    Examples
    --------
    """

    # Allocate a new view
    b = a

    # Set the relevant update conditions for the new view
    for (dim, stride) in dim_stride_tuples:
        _bh.slide_view(b, dim, stride)
    return b
