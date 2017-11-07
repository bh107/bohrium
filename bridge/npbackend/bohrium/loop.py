"""
Bohrium Loop
============
"""

from .target import runtime_flush, runtime_flush_count


def loop(func, niters, *args, **kwargs):
    """Calls the `func` `niters` times with the `*args` and `**kwargs` as argument

    Parameters
    ----------
    func : function
        The function to run in each iterations. `func` can take any argument but cannot return anything.
    niters: int
        Number of iterations in the loop (number of times `func` is called)
    *args, **kwargs : list and dict
        The arguments to `func`

    Notes
    -----
    `func` can only use operations supported natively in Bohrium.

    Examples
    --------
    In the following, `loop_body` is called 5 times:

        def loop_body(a, b):
            b += a * b
        a = M.arange(10)
        res = M.ones_like(a)
        M.loop_body(kernel, 5, a, res)
    """

    runtime_flush()
    flush_count = runtime_flush_count()
    func(*args, **kwargs)
    if flush_count != runtime_flush_count():
        raise RuntimeError("Invalid function: the looped function contains operations not support "
                           "by Bohrium or is simply too big!")
    runtime_flush(niters)

