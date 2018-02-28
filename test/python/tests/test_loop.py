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


np_for_loop_src = """
def for_loop(loop_body, niters, *args, **kwargs):
    args += (0,)
    for i in range(0, niters):
        args = args[:-1] + (i,)
        res = loop_body(*args, **kwargs)
    return res
"""


class test_for_loop_view:
    """ Test a of sliding two views with a for loop"""
    def init(self):
        cmd1 = np_for_loop_src + """
def kernel(a,b,i):
    c = a[i:i+1]
    c += b[i:i+1, i:i+1]
    return a

b = M.ones((20, 5))
b[::2, ::2] += 1
b[1::2, 1::2] += 1
res = M.zeros((5, 1))
"""
        cmd2 = np_for_loop_src + """
def kernel(a, b):
    c = bh.slide_view(a[0:1], [(0,1)])
    c += bh.slide_view(b[0:1,0:1], [(0,1), (1,1)])

b = M.ones((20, 5))
b[::2, ::2] += 1
b[1::2, 1::2] += 1
res = M.zeros((5, 1))
"""

        yield (cmd1, cmd2, 5)

    def test_func(self, args):
        """Test of the do_while function"""
        (cmd1, cmd2, niter) = args

        return (cmd1 + "for_loop(kernel, %s, res, b)" % (niter), cmd2 + "M.for_loop(kernel, %s, res, b)" % (niter))
