"""
The Computation Backend

"""
from .. import bhc
from .._util import dtype_name
import numpy as np
import mmap
import time
import ctypes
import backend
import os

VCACHE_SIZE = int(os.environ.get("VCACHE_SIZE", 10))
vcache = []

class base(backend.base):
    """base array handle"""
    def __init__(self, size, dtype):
        super(base, self).__init__(size, dtype)
        size *= dtype.itemsize
        for i, (s,m) in enumerate(vcache):
            if s == size:
                self.mmap = m
                vcache.pop(i)
                return
        self.mmap = mmap.mmap(-1, size, mmap.MAP_PRIVATE)
    def __str__(self):
        return "<base memory at %s>"%self.mmap
    def __del__(self):
        if len(vcache) < VCACHE_SIZE:
            vcache.append((self.size*self.dtype.itemsize, self.mmap))

class view(backend.view):
    """array view handle"""
    def __init__(self, ndim, start, shape, stride, base):
        super(view, self).__init__(ndim, start, shape, stride, base)
        buf = np.frombuffer(self.base.mmap, dtype=self.dtype, offset=self.start)
        self.ndarray = np.lib.stride_tricks.as_strided(buf, shape, self.stride)

def views2numpy(views):
    ret = []
    for v in views:
        if isinstance(v, view):
            ret.append(v.ndarray)
        else:
            ret.append(v)
    return ret

def get_data_pointer(ary, allocate=False, nullify=False):
    return ary.ndarray.ctypes.data

def set_bhc_data_from_ary(self, ary):
    d = get_data_pointer(self, allocate=True, nullify=False)
    ctypes.memmove(d, ary.ctypes.data, ary.dtype.itemsize * ary.size)

def ufunc(op, *args):
    """Apply the 'op' on args, which is the output followed by one or two inputs"""
    args = views2numpy(args)
    if op.info['name'] == "identity":
        exec("args[0][...] = args[1][...]")
    else:
        f = eval("np.%s"%op.info['name'])
        f(*args[1:], out=args[0])

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

    (out, in1, in2) = views2numpy((out, in1, in2))
    if name == "matmul":
        out[:] = np.dot(in1, in2)
    else:
        raise NotImplementedError("The current runtime system does not support "
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
    pass
