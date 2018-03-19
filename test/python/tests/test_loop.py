np_dw_loop_src = """
def do_while(func, niters, *args, **kwargs):
    import sys
    i = 0
    if niters is None:
        niters = sys.maxsize
    while i < niters:
        cond = func(*args, **kwargs)
        if cond is not None and not cond:
            break
        i += 1
"""


class test_loop_fixed:
    """ Test loop with fixed number of iterations"""
    def init(self):
        cmd = np_dw_loop_src + """
def kernel(a, b):
    b += a * b

a = M.arange(10);
res = M.ones_like(a)

"""
        yield (cmd)

    def test_func(self, cmd):
        """Test of the loop function"""
        return (cmd + "do_while(kernel, 5, a, res)", cmd + "M.do_while(kernel, 5, a, res)")


class test_loop_cond:
    """ Test loop with a condition variable"""
    def init(self):
        cmd = np_dw_loop_src + """
def kernel(a, b):
    b += a * b
    return M.sum(b) < 10000

a = M.arange(10);
res = M.ones_like(a)

"""
        yield (cmd, 1000000)
        yield (cmd, 3)
        yield (cmd, None)

    def test_func(self, args):
        """Test of the do_while function"""
        (cmd, niter) = args

        return (cmd + "do_while(kernel, %s, a, res)" % (niter), cmd + "M.do_while(kernel, %s, a, res)" % (niter))


np_dw_loop_slide_src = """
def do_while_i(func, niters, *args, **kwargs):
    import sys
    i = 0
    if niters is None:
        niters = sys.maxsize
    args += (0,)
    while i < niters:
        args = args[:-1] + (i,)
        cond = func(*args, **kwargs)
        if cond is not None and not cond:
            break
        i += 1
"""

class test_loop_sliding_view:
    """ Test a of sliding two views with a for loop"""
    def init(self):
        cmd1 = np_dw_loop_slide_src + \
"""
def kernel(a,b,i):
    a[i] += b[i, i]

b = M.ones((20, 5))
b[::2, ::2] += 1
b[1::2, 1::2] += 1
res = M.zeros((5, 1))
"""
        cmd2 = np_dw_loop_src + \
"""
def kernel(a, b):
    i = get_iterator()
    a[i] += b[i, i]

b = M.ones((20, 5))
b[::2, ::2] += 1
b[1::2, 1::2] += 1
res = M.zeros((5, 1))
"""
        yield (cmd1, cmd2, 5)

    def test_func(self, args):
        """Test of the do_while function"""
        (cmd1, cmd2, niter) = args

        return (cmd1 + "do_while_i(kernel, %s, res, b)" % (niter), cmd2 + "M.do_while(kernel, %s, res, b)" % (niter))
