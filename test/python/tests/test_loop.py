import util

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
    """ Test a of sliding two views within the do_while loop.
    The calculation results in the triangular numbers."""
    def init(self):
        cmd1 = np_dw_loop_slide_src + \
"""
iter = %s

def kernel(a,i):
    a[i+1] += a[i]
res = M.arange(1,iter+2)
do_while_i(kernel, iter, res)
"""
        cmd2 = np_dw_loop_src + \
"""
iter = %s

def kernel(a):
    i = get_iterator()
    a[i+1] += a[i]
res = M.arange(1,iter+2)
M.do_while(kernel, iter, res)
"""
        yield (cmd1, cmd2, 10)

    def test_func(self, args):
        (cmd1, cmd2, niter) = args
        return (cmd1 % (niter), cmd2 % (niter))


class test_loop_one_and_two_dimensional_sliding_views:
    """Test of sliding two views with a for loop. One view is one-dimensional, while the other is two-dimensional"""
    def init(self):
        cmd1 = np_dw_loop_slide_src + \
"""
iter = %s

def kernel(a,b,i):
    a[i] += b[i, i]

b = M.ones((20, iter))
b[::2, ::2] += 1
b[1::2, 1::2] += 1
res = M.zeros((iter, 1))
do_while_i(kernel, iter, res, b)
"""
        cmd2 = np_dw_loop_src + \
"""
iter = %s

def kernel(a, b):
    i = get_iterator()
    a[i] += b[i, i]

b = M.ones((20, iter))
b[::2, ::2] += 1
b[1::2, 1::2] += 1
res = M.zeros((iter, 1))
M.do_while(kernel, iter, res, b)
"""
        yield (cmd1, cmd2, 5)

    def test_func(self, args):
        (cmd1, cmd2, niter) = args
        return (cmd1 % (niter), cmd2 % (niter))


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
        (cmd1, cmd2, niter) = args
        return (cmd1 % (niter), cmd2 % (niter))


class test_loop_sliding_view_negative_index_3d:
    """Test of negative sliding in a 3-dimensional view"""
    def init(self):
        cmd1 = np_dw_loop_slide_src + \
"""
iter = %s

def kernel(a,i):
    a[-i, -i, -i] += 1

res = M.zeros((iter,iter,iter))
do_while_i(kernel, iter, res)
"""
        cmd2 = np_dw_loop_src + \
"""
iter = %s

def kernel(a):
    i = get_iterator()
    a[-i, -i, -i] += 1

res = M.zeros((iter,iter,iter))
M.do_while(kernel, iter, res)
"""
        yield (cmd1, cmd2, 3)

    def test_func(self, args):
        (cmd1, cmd2, niter) = args
        return (cmd1 % (niter), cmd2 % (niter))


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
except M.loop.IteratorOutOfBounds:
    pass

try:
    M.do_while(kernel_out_of_bounds_underflow, len(res), dummy)
    failure = True
except M.loop.IteratorOutOfBounds:
    pass

if not failure:
    M.do_while(kernel, iter, res)
"""
        yield (cmd1, cmd2, 5)

    def test_func(self, args):
        """Test exceptions of underflow and overflow"""
        (cmd1, cmd2, niter) = args
        return (cmd1 + "do_while_i(kernel, %s, res)" % (niter), cmd2 % (niter))


class test_3d_grid:
    """Test iterating through a 3d grid"""
    def init(self):
        cmd1 = np_dw_loop_slide_src + \
"""
iter = (%s, %s, %s)

res = np.zeros(iter)
counter = np.zeros(1)
for i in range(iter[0]):
    for j in range(iter[1]):
        for k in range(iter[2]):
            counter += 1
            res[i,j,k] += counter
"""

        cmd2 = np_dw_loop_src + \
"""
iter = (%s, %s, %s)
def kernel(res, counter):
    i, j, k = get_grid(*iter)
    counter += 1
    res[i,j,k] += counter

res = bh.zeros(iter)
counter = bh.zeros(1)

M.do_while(kernel, iter[0]*iter[1]*iter[2], res, counter)
"""
        yield (cmd1, cmd2, (4,4,4))

    def test_func(self, args):
        """Test exceptions of underflow and overflow"""
        (cmd1, cmd2, niter) = args
        return (cmd1 % niter, cmd2 % niter)


class test_dynamic_vector_broadcast:
    """Test broadcasting a single value to a vector with a shape that
    changes between each iteration"""
    def init(self):
        cmd1 = \
"""
iter = %s

res = M.zeros(iter)
b = M.arange(iter)+1

for i in range(1,iter+1):
    res[:-i] += b[i-1]
"""

        cmd2 = \
"""
iter = %s

def loop_body(res, b):
    i = get_iterator(1)
    res[:-i] += b[i-1]

res = M.zeros(iter)
b = M.arange(iter)+1
M.do_while(loop_body, iter, res, b)
"""
        yield (cmd1, cmd2, 15)

    def test_func(self, args):
        (cmd1, cmd2, niter) = args
        return (cmd1 % niter, cmd2 % niter)


class test_dynamic_tensor_broadcast:
    """Test broadcasting a single value to a tensor with a shape that
    changes between each iteration"""
    def init(self):
        cmd1 = \
"""
iter = %s

res = M.zeros((iter,iter,iter))
b = M.arange(iter)+1

for i in range(1,iter+1):
    res[:-i,:-i,:-i] += b[i-1]
"""

        cmd2 = \
"""
iter = %s

def loop_body(res, b):
    i = get_iterator(1)
    res[:-i,:-i,:-i] += b[i-1]

res = M.zeros((iter,iter,iter))
b = M.arange(iter)+1
M.do_while(loop_body, iter, res, b)
"""
        yield (cmd1, cmd2, 15)

    def test_func(self, args):
        (cmd1, cmd2, niter) = args
        return (cmd1 % niter, cmd2 % niter)


class test_gaussian_elimination:
    """Test of gaussian elimination on a 10 by 10 matrix (equation system)"""
    def init(self):
        cmd = "R = bh.random.RandomState(42); S = R.random((10,10), dtype=np.float, bohrium=BH); "
        cmd1 = cmd + \
"""
for c in range(1, S.shape[0]):
    S[c:, c - 1:] -= (S[c:,c-1:c] / S[c-1:c, c-1:c]) * S[c-1:c,c-1:]

S /= np.diagonal(S)[:, None]
res = S
"""
        cmd2 = cmd + \
"""
def loop_body(S):
    c = get_iterator(1)
    S[c:, c - 1:] -= (S[c:,c-1:c] / S[c-1:c, c-1:c]) * S[c-1:c,c-1:]

M.do_while(loop_body, S.shape[0]-1, S)
S /= np.diagonal(S)[:, None]
res = S
"""
        yield (cmd1, cmd2)

    def test_func(self, args):
        (cmd1, cmd2) = args
        return (cmd1, cmd2)


class test_nested_dynamic_view:
    """Test of a nested dynamic view"""
    def init(self):
        cmd1 = \
"""
iter = %s

res = M.zeros(iter**2)
for i in range(iter):
    a = res[i:i+iter]
    a[i] += 1
"""

        cmd2 = \
"""
iter = %s

res = M.zeros(iter**2)
def loop_body(res):
    i = get_iterator()
    a = res[i:i+iter]
    a[i] += 1

M.do_while(loop_body, iter, res)
"""
        yield (cmd1, cmd2, 15)

    def test_func(self, args):
        (cmd1, cmd2, niter) = args
        return (cmd1 % niter, cmd2 % niter)


class test_advanced_nested_dynamic_view:
    """Test of three nested views"""
    def init(self):
        cmd1 = \
"""
a = M.zeros(32)
for i in range(2):
    b = a[16*i:16*(i+1)-1:2]
    b += 1
    c = b[4*i:4*(i+1)-1:2]
    c += 1
    c[i] += 1
res = a
"""

        cmd2 = \
"""
def loop_body(a):
    i = get_iterator()
    b = a[16*i:16*(i+1)-1:2]
    b += 1
    c = b[4*i:4*(i+1)-1:2]
    c += 1
    c[i] += 1

res = M.zeros(32)
M.do_while(loop_body, 2, res)
"""
        yield (cmd1, cmd2)

    def test_func(self, args):
        (cmd1, cmd2) = args
        return (cmd1, cmd2)


class test_do_while_convolution:
    """Test of a convolution that takes the average of a 3 by 3 window"""
    def init(self):
        cmd = "R = bh.random.RandomState(42); S = R.random((10,10), dtype=np.float, bohrium=BH); "
        cmd1 = cmd + \
"""
a = M.zeros((12,12))
a[1:-1,1:-1] += S
b = M.zeros((10,10))
for i in range(10):
    for j in range(10):
        point = b[i:i+1, j:j+1]
        window = a[i:i+3, j:j+3]
        point += M.mean(window)
res = b
"""
        cmd2 = cmd +\
"""
def convolution(a, b):
    i, j = get_grid(10,10)
    point = b[i, j]
    window = a[i:i+3, j:j+3]
    point += M.mean(window)

a = M.zeros((12,12))
a[1:-1,1:-1] += S
b = M.zeros((10,10))
M.do_while(convolution, b.shape[0] * b.shape[1], a, b)
res = b
"""
        yield (cmd1, cmd2)

    def test_func(self, args):
        (cmd1, cmd2) = args
        return (cmd1, cmd2)


class test_temp_arrays_with_changing_shape:
    """Test of temporary arrays with changing shape"""
    def init(self):
        cmd1 = \
"""
iter = %s

a = M.ones(iter*5)
res = M.zeros(iter)

for i in range(iter):
    b = a[i:iter]
    c = a[i+iter:2*iter]
    d = a[i+2*iter:3*iter]
    e = a[i+3*iter:4*iter]
    res[i:] += b+c+d+e
"""
        cmd2 = \
"""
iter = %s

def loop_body(a, res):
    i = get_iterator()
    b = a[i:iter]
    c = a[i+iter:2*iter]
    d = a[i+2*iter:3*iter]
    e = a[i+3*iter:4*iter]
    res[i:] += b+c+d+e

a = M.ones(iter*5)
res = M.zeros(iter)
M.do_while(loop_body, iter, a, res)
"""
        yield (cmd1, cmd2, 5)

    def test_func(self, args):
        (cmd1, cmd2, niter) = args
        return (cmd1 % niter, cmd2 % niter)


class test_loop_illegal_iterator_mix:
    """Test of mixing iterators within a grid illegally"""
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

def iterator_mix1(a):
    i, j = get_grid(5,5)
    b = a[i:i+5,j:j+5]
    c = b[j, i]
    c += 1

def iterator_mix2(a):
    i, j = get_grid(5,5)
    k, l = get_grid(4,4)
    b = a[i:i+5]
    c = b[k]
    c += 1

def kernel(a):
    i = get_iterator()
    a[i] += 1

a = bh.zeros((10,10))

dummy = M.zeros(iter)
res   = M.zeros(iter)
failure = False

try:
    M.do_while(iterator_mix1, 2, a)
    failure = True
except M.loop.IteratorIllegalDepth:
    pass

try:
    M.do_while(iterator_mix2, 2, a)
    failure = True
except M.loop.IteratorIllegalDepth:
    pass

if not failure:
    M.do_while(kernel, iter, res)
"""
        yield (cmd1, cmd2, 5)

    def test_func(self, args):
        """Test exceptions of underflow and overflow"""
        (cmd1, cmd2, niter) = args
        return (cmd1 + "do_while_i(kernel, %s, res)" % (niter), cmd2 % (niter))
