"""
The Computation Backend

"""
import bhc
import numpy as np
from _util import dtype_name
import mmap
import time

VCACHE_SIZE = 10
vcache = []

class base(object):
    """base array handle"""
    def __init__(self, size, dtype):
        self.size = size
        size *= dtype.itemsize
        self.dtype = dtype
        for i, (s,m) in enumerate(vcache):
            if s == size:
                self.mmap = m
                vcache.pop(i)
#                print "create (hit)", self
                return
        self.mmap = mmap.mmap(-1, size)
#        print "create (miss)", self
    def __str__(self):
        return "<base memory at %s>"%self.mmap
    def __del__(self):
#        print "del", self
        if len(vcache) < VCACHE_SIZE:
            vcache.append((self.size*self.dtype.itemsize, self.mmap))

class view(object):
    """array view handle"""
    def __init__(self, ndim, start, shape, stride, base, dtype):
        assert(dtype == base.dtype)
        self.ndim = ndim
        self.start = start
        self.shape = shape
        self.stride = stride
        self.base = base
        self.dtype = dtype
        buf = np.frombuffer(self.base.mmap, dtype=dtype, offset=start*dtype.itemsize)
        stride = [x * dtype.itemsize for x in stride]
        self.ndarray = np.lib.stride_tricks.as_strided(buf, shape, stride)

def new_empty(size, dtype):
    """Return a new empty base array"""
    return base(size, dtype)

def new_view(ndim, start, shape, stride, base, dtype):
    """Return a new view that points to 'base'"""
    return view(ndim, start, shape, stride, base, dtype)

def views2numpy(views):
    ret = []
    for v in views:
        if isinstance(v, view):
            ret.append(v.ndarray)
        else:
            ret.append(v)
    return ret

def get_data_pointer(ary, allocate=False, nullify=False):
#    print "get_data_pointer", type(ary.ndarray.base),
    return ary.ndarray.ctypes.data

#t_ufunc = 0

def ufunc(op, *args):
    """Apply the 'op' on args, which is the output followed by one or two inputs"""
#    global t_ufunc
#    print "ufunc: %s "%op.info['name'], [(x.base if isinstance(x, base) else x)  for x in args]
    args = views2numpy(args)
#    t1 = time.time()
    if op.info['name'] == "identity":
        exec "args[0][...] = args[1][...]"
    else:
        f = eval("np.%s"%op.info['name'])
        f(*args[1:], out=args[0])
#    t2 = time.time()
#    t_ufunc += t2-t1

def reduce(op, out, a, axis):
    """reduce 'axis' dimension of 'a' and write the result to out"""

    f = eval("np.%s.reduce"%op.info['name'])
    (a, out) = views2numpy((a, out))
    if a.ndim == 1:
        keepdims = True
    else:
        keepdims = False
    f(a, axis=axis, out=out, keepdims=keepdims)

def accumulate(op, out, a, axis):
    """accumulate 'axis' dimension of 'a' and write the result to out"""

    f = eval("np.%s.accumulate"%op.info['name'])
    (a, out) = views2numpy((a, out))
    if a.ndim == 1:
        keepdims = True
    else:
        keepdims = False
    f(a, axis=axis, out=out, keepdims=keepdims)

def extmethod(name, out, in1, in2):
    """Apply the extended method 'name' """

    f = eval("bhc.bh_multi_array_extmethod_%s_%s_%s"%(dtype_name(out),\
              dtype_name(in1), dtype_name(in2)))
    ret = bhc_exec(f, name, out, in1, in2)
    if ret != 0:
        raise RuntimeError("The current runtime system does not support "
                           "the extension method '%s'"%name)

def range(size, dtype):
    """create a new array containing the values [0:size["""
    raise NotImplementedError()
    return np.arange((size,), dtype=dtype)

def random123(size, start_index, key):
    """Create a new random array using the random123 algorithm.
    The dtype is uint64 always."""
    raise NotImplementedError()
    return np.random.random(size)


import atexit
@atexit.register
def shutdown():
#    print "ufunc:", t_ufunc
#    print "vcache size at exit: %d"%len(vcache)
    pass
