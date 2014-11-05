"""
numexpr as the target for npbackend.
"""
from __future__ import print_function
from .. import bhc
from .._util import dtype_name
import numpy as np
import time
import numexpr
import os
import ctypes
from . import target_numpy

class Base(target_numpy.Base):
    """base array handle"""
    pass


class View(target_numpy.View):
    """array view handle"""
    pass

def views2numpy(views):
    """Returns the ndarray associated with the given View."""
    ret = []
    for view in views:
        if isinstance(view, View):
            ret.append(view.ndarray)
        else:
            ret.append(view)
    return ret

def get_data_pointer(ary, allocate=False, nullify=False):
    """Retrieve the data-pointer associated with 'ary'."""
    return ary.ndarray.ctypes.data

def set_bhc_data_from_ary(self, ary):
    """Assign the data from 'ary' to 'self'."""
    ptr = get_data_pointer(self, allocate=True, nullify=False)
    ctypes.memmove(ptr, ary.ctypes.data, ary.dtype.itemsize * ary.size)

# Setup numexpr
numexpr.set_num_threads(int(os.getenv('OMP_NUM_THREADS', 1)))
print("using numexpr target with %d threads" % 
      int(os.getenv('OMP_NUM_THREADS', 1))
)
UFUNC_CMDS = {
    'identity': "i1",
    'add':      "i1 + i2",
    'subtract': "i1 - i2",
    'multiply': "i1 * i2",
    'divide':   "i1 / i2",
    'power':    "i1**i2",
    'absolute': "abs(i1)",
    'sqrt':     "sqrt(i1)"
}

def ufunc(op, *args):
    """Apply the 'op' on args, which is the output followed by one or two inputs"""
    args = views2numpy(args)
    i1 = args[1]
    if len(args) > 2:
        i2 = args[2]

    if op.info['name'] in UFUNC_CMDS:
        numexpr.evaluate(
            UFUNC_CMDS[op.info['name']],
            out=args[0],
            casting='unsafe'
        )
    else:
        print("WARNING: ufunc '%s' not compiled" % op.info['name'])
        func = eval("np.%s" % op.info['name'])
        func(*args[1:], out=args[0])

def reduce(op, out, ary, axis):
    """reduce 'axis' dimension of 'a' and write the result to out"""

    (ary, out) = views2numpy((ary, out))
    if op.info['name'] == 'add':
        numexpr.evaluate("sum(ary, %d)" % axis, out=out, casting='unsafe')
    elif op.info['name'] == 'multiply':
        numexpr.evaluate("prod(ary, %d)" % axis, out=out, casting='unsafe')
    else:
        print ("WARNING: reduce '%s' not compiled" % op.info['name'])
        func = eval("np.%s.reduce" % op.info['name'])
        if ary.ndim == 1:
            keepdims = True
        else:
            keepdims = False
        func(ary, axis=axis, out=out, keepdims=keepdims)

def accumulate(op, out, ary, axis):
    """accumulate 'axis' dimension of 'a' and write the result to out"""

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
