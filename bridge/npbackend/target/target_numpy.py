"""
The Computation Backend

"""
from .. import bhc
from .._util import dtype_name
import numpy as np
import mmap
import time
import ctypes
from . import target
import os

VCACHE = []
VCACHE_SIZE = int(os.environ.get("VCACHE_SIZE", 10))

class Base(target.Base):
    """base array handle"""

    def __init__(self, size, dtype):
        super(Base, self).__init__(size, dtype)
        size *= dtype.itemsize
        for i, (vc_size, vc_mem) in enumerate(VCACHE):
            if vc_size == size:
                self.mmap = vc_mem
                VCACHE.pop(i)
                return

        self.mmap = mmap.mmap(-1, size, mmap.MAP_PRIVATE)

    def __str__(self):
        return "<base memory at %s>"%self.mmap

    def __del__(self):
        if len(VCACHE) < VCACHE_SIZE:
            VCACHE.append((self.size*self.dtype.itemsize, self.mmap))

class View(target.View):
    """array view handle"""

    def __init__(self, ndim, start, shape, strides, base):
        super(View, self).__init__(ndim, start, shape, strides, base)
        buf = np.frombuffer(self.base.mmap, dtype=self.dtype, offset=self.start)
        self.ndarray = np.lib.stride_tricks.as_strided(buf, shape, self.strides)

def views2numpy(views):
    """Extract the ndarray from the view."""

    ret = []
    for view in views:
        if isinstance(view, View):
            ret.append(view.ndarray)
        else:
            ret.append(view)
    return ret

def get_data_pointer(ary, allocate=False, nullify=False):
    """
    Extract the data-pointer from the given View (ary).

    :param target_numpy.View ary: The View to extract the ndarray form.
    :returns: Pointer to data associated with the 'ary'.
    :rtype: ctypes pointer
    """
    return ary.ndarray.ctypes.data

def set_bhc_data_from_ary(self, ary):
    ptr = get_data_pointer(self, allocate=True, nullify=False)
    ctypes.memmove(ptr, ary.ctypes.data, ary.dtype.itemsize * ary.size)

def ufunc(op, *args):
    """Apply the 'op' on args, which is the output followed by one or two inputs"""

    args = views2numpy(args)
    if op.info['name'] == "identity":
        exec("args[0][...] = args[1][...]")
    else:
        func = eval("np.%s" % op.info['name'])
        func(*args[1:], out=args[0])

def reduce(op, out, ary, axis):
    """reduce 'axis' dimension of 'ary' and write the result to out"""

    func = eval("np.%s.reduce" % op.info['name'])
    (ary, out) = views2numpy((ary, out))
    if ary.ndim == 1:
        keepdims = True
    else:
        keepdims = False
    func(ary, axis=axis, out=out, keepdims=keepdims)

def accumulate(op, out, ary, axis):
    """accumulate 'axis' dimension of 'ary' and write the result to out"""

    func = eval("np.%s.accumulate" % op.info['name'])
    (ary, out) = views2numpy((ary, out))
    if ary.ndim == 1:
        keepdims = True
    else:
        keepdims = False
    func(ary, axis=axis, out=out, keepdims=keepdims)

def extmethod(name, out, in1, in2):
    """Apply the extended method 'name' """

    (out, in1, in2) = views2numpy((out, in1, in2))
    if name == "matmul":
        out[:] = np.dot(in1, in2)
    else:
        raise NotImplementedError("The current runtime system does not support "
                                  "the extension method '%s'" % name)

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
