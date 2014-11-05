"""
Chapel as target for npbackend.
"""
from .. import bhc
from .._util import dtype_name
import numpy as np
import mmap
import time
import ctypes
import target
import pprint
import pych
from pych.array_ops import *
import os

def views2numpy(views):
    ret = []
    for v in views:
        if isinstance(v, view):
            ret.append(v.ndarray)
        else:
            ret.append(v)
    return ret

class base(target.base):
    """base array handle"""

    def __init__(self, size, dtype):
        super(base, self).__init__(size, dtype, mem_protected=False)
        self.ndarray = np.ones(size, dtype=dtype)

    def __del__(self):
        del self.ndarray

class view(target.view):
    """array view handle"""
    def __init__(self, ndim, start, shape, stride, base):
        super(view, self).__init__(ndim, start, shape, stride, base)
        self.ndarray = self.base.ndarray
        self.mem_protected = False

def get_data_pointer(ary, allocate=False, nullify=False):
    print "get_data_pointer"
    return ary.ndarray.ctypes.data

def set_bhc_data_from_ary(self, ary):
    print "set_bhc_data_from_ary"

def ufunc(op, *args):
    """Apply the 'op' on args, which is the output followed by one or two inputs"""

    (res, in1) = views2numpy(args)
    in1 = int(in1)
    pych_ewise_assign(in1, res)

    print "ufunc", op
    pprint.pprint(op)
    pprint.pprint(args)

def reduce(op, out, a, axis):
    """reduce 'axis' dimension of 'a' and write the result to out"""
   
    (a, out) = views2numpy((a, out))
    pych_reduce_add(a, axis, out)

    print "reduce"
    pprint.pprint(op)
    pprint.pprint(out)
    pprint.pprint(a)
    pprint.pprint(axis)

def accumulate(op, out, a, axis):
    """accumulate 'axis' dimension of 'a' and write the result to out"""

    print "accumulate"
    pprint.pprint(op)
    pprint.pprint(out)
    pprint.pprint(a)
    pprint.pprint(axis)

def extmethod(name, out, in1, in2):
    """Apply the extended method 'name' """

    print "extmethod"
    pprint.pprint(name)
    pprint.pprint(out)
    pprint.pprint(in1)
    pprint.pprint(in2)

def range(size, dtype):
    """create a new array containing the values [0:size["""

    print "range"

def random123(size, start_index, key):
    """Create a new random array using the random123 algorithm.
    The dtype is uint64 always."""

    print "random123"

    
import atexit
@atexit.register
def shutdown():
    pass
