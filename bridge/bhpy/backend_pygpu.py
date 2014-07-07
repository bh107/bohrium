"""
The Computation Backend

"""
import bhc
import numpy as np
from _util import dtype_name
import mmap
import time
import os
import pygpu
from pygpu.array import gpuarray as elemary

cxt = pygpu.init("opencl0:0")
#cxt = pygpu.init("cuda0")
pygpu.set_default_context(cxt)

VCACHE_SIZE = 0
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
        self.clary = pygpu.empty((size,), dtype=dtype, cls=elemary)
        self.mmap = mmap.mmap(-1, size)
#        print "create (miss)", self
    def __str__(self):
        return "<base memory at %s>"%self.mmap
    def __del__(self):
#        print "del", self, self.clary
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

        self.clary = pygpu.gpuarray.from_gpudata(base.clary.gpudata, offset=start*dtype.itemsize, dtype=dtype, shape=shape, strides=stride, writable=True, base=base.clary, cls=elemary)
#        print "view", type(self.clary), self.clary

def new_empty(size, dtype):
    """Return a new empty base array"""
    return base(size, dtype)

def new_view(ndim, start, shape, stride, base, dtype):
    """Return a new view that points to 'base'"""
    return view(ndim, start, shape, stride, base, dtype)

def views2clary(views):
    ret = []
    for v in views:
        if isinstance(v, view):
            ret.append(v.clary)
        else:
            ret.append(v)
    return ret

def get_data_pointer(ary, allocate=False, nullify=False):
#    print "get_data_pointer", type(ary.ndarray.base)
    ary.ndarray[:] = np.asarray(ary.clary)
    return ary.ndarray.ctypes.data

def set_bhc_data_from_ary(self, ary):
#    print "set_bhc_data_from_ary", type(self), type(ary)
    self.clary[:] = pygpu.asarray(ary)


ufunc_cmds = {'identity' : "i1",
              'add' : "i1+i2",
              'subtract' : "i1-i2",
              'multiply' : "i1*i2",
              'divide' : "i1/i2",
              'power' : "i1**i2",
              'absolute' : "abs(i1)",
              'sqrt' : "sqrt(i1)",
              }

def ufunc(op, *args):
    """Apply the 'op' on args, which is the output followed by one or two inputs"""
    #print "ufunc: %s "%op.info['name'], [(x.base if isinstance(x, base) else x)  for x in args]
    args = views2clary(args)

    out=args[0]
    i1=args[1];
    if len(args) > 2:
        i2=args[2]

    if op.info['name'] == "identity":
        if out.base is i1.base:#PyGPU does not support inplace copy (it seems)
            i1 = i1.copy()
        out[:] = i1
    elif op.info['name'] in ufunc_cmds:
        cmd = "out[:] = %s"%ufunc_cmds[op.info['name']]
        exec cmd
    else:
        raise NotImplementedError()

def reduce(op, out, a, axis):
    """reduce 'axis' dimension of 'a' and write the result to out"""

    (a, out) = views2clary((a, out))
    if op.info['name'] == 'add':
        rfun = a.sum
    elif op.info['name'] == 'multiply':
        rfun = a.prod
    else:
        raise NotImplementedError()
    if a.ndim == 1:
        out[:] = rfun(axis=axis)
    else:
        rfun(axis=axis, out=out)

def accumulate(op, out, a, axis):
    """accumulate 'axis' dimension of 'a' and write the result to out"""

    raise NotImplementedError()

def extmethod(name, out, in1, in2):
    """Apply the extended method 'name' """

    raise NotImplementedError()

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
