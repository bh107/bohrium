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


class test_loop_faculty_function_using_sliding_views:
    """ Test a of sliding two views with a for loop"""
    def init(self):
        cmd1 = np_dw_loop_slide_src + \
"""
def kernel(a,i):
    a[i+1] += a[i]
res = M.arange(1,6)
"""
        cmd2 = np_dw_loop_src + \
"""
def kernel(a):
    i = get_iterator()
    a[i+1] += a[i]
res = M.arange(1,6)
"""
        yield (cmd1, cmd2, 4)

    def test_func(self, args):
        """Test of the loop-based faculty function"""
        (cmd1, cmd2, niter) = args

        return (cmd1 + "do_while_i(kernel, %s, res)" % (niter), cmd2 + "M.do_while(kernel, %s, res)" % (niter))


class test_loop_one_and_two_dimensional_sliding_views:
    """Test of sliding two views with a for loop. One view is one-dimensional, while the other is two-dimensional"""
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


class test_loop_sliding_view_index_switch_negative_positive:
    """Test of a sliding view that goes from a negative to a positive index (and vice versa)"""
    def init(self):
        cmd1 = np_dw_loop_slide_src + \
"""
iter = %s
def kernel(a,i):
    a[i] += 1

def kernel2(a,i):
    a[-i] += 1

res = M.zeros(iter)
do_while_i(kernel, iter, res)
do_while_i(kernel2, iter, res)
"""
        cmd2 = np_dw_loop_src + \
"""
iter = %s
def kernel(a):
    i = get_iterator(-2)
    a[i] += 1

def kernel2(a):
    i = get_iterator(-2)
    a[-i] += 1

res = M.zeros(iter)
M.do_while(kernel, iter, res)
M.do_while(kernel2, iter, res)
"""
        yield (cmd1, cmd2, 5)

    def test_func(self, args):
        """Test of the do_while function"""
        (cmd1, cmd2, niter) = args

        return (cmd1 % (niter), cmd2 % (niter))


class test_loop_sliding_view_negative_index_3d:
    """Test of negative sliding in a 3-dimensional view"""
    def init(self):
        cmd1 = np_dw_loop_slide_src + \
"""
def kernel(a,i):
    a[-i, -i, -i] += 1

res = M.zeros((3,3,3))
"""
        cmd2 = np_dw_loop_src + \
"""
def kernel(a):
    i = get_iterator()
    a[-i, -i, -i] += 1

res = M.zeros((3,3,3))
"""
        yield (cmd1, cmd2, 3)

    def test_func(self, args):
        """Test of the do_while function"""
        (cmd1, cmd2, niter) = args

        return (cmd1 + "do_while_i(kernel, %s, res)" % (niter), cmd2 + "M.do_while(kernel, %s, res)" % (niter))


class test_loop_sliding_view_out_of_bounds:
    """Test a of error checks when sliding out of bounds"""
    def init(self):
        cmd1 = np_dw_loop_slide_src + \
"""
def kernel(a,i):
    a[i] += 1

res = M.zeros(5)
"""
        cmd2 = np_dw_loop_src + \
"""
iter = %s

def kernel_out_of_bounds_overflow(a):
    i = get_iterator(1)
    a[i] += 1

def kernel_out_of_bounds_underflow(a):
    i = get_iterator(2)
    a[-i] += 1

def kernel(a):
    i = get_iterator()
    a[i] += 1

dummy = M.zeros(iter)
res   = M.zeros(iter)
failure = False

try:
    M.do_while(kernel_out_of_bounds_overflow, len(res), dummy)
    failure = True
except M.iterator.IteratorOutOfBounds:
    pass

try:
    M.do_while(kernel_out_of_bounds_underflow, len(res), dummy)
    failure = True
except M.iterator.IteratorOutOfBounds:
    pass

if not failure:
    M.do_while(kernel, iter, res)
"""
        yield (cmd1, cmd2, 5)

    def test_func(self, args):
        """Test exceptions of underflow and overflow"""
        (cmd1, cmd2, niter) = args
        return (cmd1 + "do_while_i(kernel, %s, res)" % (niter), cmd2 % (niter))


class test_loop_sliding_change_shape:
    """Test detecting the view changing shape between iterations"""
    def init(self):
        cmd1 = np_dw_loop_slide_src + \
"""
def kernel(a,i):
    a[i] += 1

res = M.zeros(5)
"""
        cmd2 = np_dw_loop_src + \
"""
iter = %s

def kernel_excp1(a):
    i = get_iterator()
    a[i:2*i] += 1

def kernel_excp2(a):
    i = get_iterator()
    a[0:i] += 1

def kernel_excp3(a):
    i = get_iterator()
    a[i:iter] += 1

def kernel_excp4(a):
    i = get_iterator()
    a[i*2:i] += 1

def kernel(a):
    i = get_iterator()
    a[i] += 1

dummy = M.zeros(iter)
res   = M.zeros(iter)
failure = False

try:
    M.do_while(kernel_excp1, iter, dummy)
    failure = True
except M.iterator.ViewShape:
    pass

try:
    M.do_while(kernel_excp2, iter, dummy)
    failure = True
except M.iterator.ViewShape:
    pass

try:
    M.do_while(kernel_excp3, iter, dummy)
    failure = True
except M.iterator.ViewShape:
    pass

try:
    M.do_while(kernel_excp4, iter, dummy)
    failure = True
except M.iterator.ViewShape:
    pass

if not failure:
    M.do_while(kernel, iter, res)
"""
        yield (cmd1, cmd2, 5)

    def test_func(self, args):
        """Test exceptions of views changing shape between iterations"""
        (cmd1, cmd2, niter) = args
        return (cmd1 + "do_while_i(kernel, %s, res)" % (niter), cmd2 % (niter))
