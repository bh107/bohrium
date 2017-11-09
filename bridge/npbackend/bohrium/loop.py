"""
Bohrium Loop
============
"""

import sys
from .target import runtime_flush, runtime_flush_count, runtime_flush_and_repeat, runtime_sync
from .bhary import get_bhc


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

    runtime_flush()
    flush_count = runtime_flush_count()
    cond = func(*args, **kwargs)
    if flush_count != runtime_flush_count():
        raise RuntimeError("Invalid function: the looped function contains operations not support "
                           "by Bohrium or is simply too big!")
    if niters is None:
        niters = sys.maxsize-1
    if cond is None:
        runtime_flush_and_repeat(niters, None)
    else:
        cond = get_bhc(cond)
        runtime_sync(cond)
        runtime_flush_and_repeat(niters, cond)
