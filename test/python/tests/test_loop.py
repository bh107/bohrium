
np_loop_src = """
def loop(func, niters, *args, **kwargs):
    for _ in range(niters):
        func(*args, **kwargs)
"""



class test_loop_fixed:
    """ Test loop with fixed number of iterations"""
    def init(self):
        cmd = "a = M.arange(10);"
        yield (cmd)

    def test_lowlevel(self, cmd):
        """Low level test without using any Python syntax sugar"""
        bh_cmd = cmd + """
def kernel(a, b):
    b += a * b
res = M.ones_like(a)
bh.flush()
kernel(a, res)
bh.flush(nrepeats=5)
"""
        np_cmd = cmd + np_loop_src + """
def kernel(a, b):
    b += a * b
res = M.ones_like(a)
loop(kernel, 5, a, res)
"""
        return (np_cmd, bh_cmd)

    def test_func(self, cmd):
        """Test of the loop function"""
        bh_cmd = cmd + """
def kernel(a, b):
    b += a * b
res = M.ones_like(a)
M.loop(kernel, 5, a, res)
"""
        np_cmd = cmd + np_loop_src + """
def kernel(a, b):
    b += a * b
res = M.ones_like(a)
loop(kernel, 5, a, res)
"""
        return (np_cmd, bh_cmd)
