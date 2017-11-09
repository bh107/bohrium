
np_loop_src = """
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
        cmd = np_loop_src + """
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
        cmd = np_loop_src + """
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