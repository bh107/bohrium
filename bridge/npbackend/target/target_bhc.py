"""
Bohrium C-backend as target for npbackend.
"""
import ctypes
import numpy
from .. import bhc
from .._util import dtype_name
from . import interface
import functools
import operator

class Base(interface.Base):
    """base array handle"""

    def __init__(self, size, dtype, bhc_obj=None):
        super(Base, self).__init__(size, dtype)
        if size == 0:
            return

        if bhc_obj is None:
            func = eval("bhc.bhc_new_A%s" % dtype_name(dtype))
            bhc_obj = _bhc_exec(func, size)
        self.bhc_obj = bhc_obj

    def __del__(self):
        if self.size == 0:
            return
        exec("bhc.bhc_destroy_A%s(self.bhc_obj)" %
             dtype_name(self.dtype)
        )

class View(interface.View):
    """array view handle"""

    def __init__(self, ndim, start, shape, strides, base):
        super(View, self).__init__(ndim, start, shape, strides, base)
        self.size = functools.reduce(operator.mul, shape, 1)
        if self.size == 0:
            return
        dtype = dtype_name(self.dtype)
        func = eval("bhc.bhc_view_A%s" % dtype)
        self.bhc_obj = func(base.bhc_obj, ndim, start, shape, strides)

    def __del__(self):
        if self.size == 0:
            return
        exec("bhc.bhc_destroy_A%s(self.bhc_obj)" %
             dtype_name(self.dtype)
        )

def _bhc_exec(func, *args):
    """execute the 'func' with the bhc objects in 'args'"""

    args = list(args)
    for i in xrange(len(args)):
        if isinstance(args[i], View):
            if not hasattr(args[i], 'bhc_obj'):
                return#Ignore zero-sized views
            args[i] = args[i].bhc_obj
    return func(*args)

def runtime_flush():
    """Flush the runtime system"""
    bhc.bhc_flush()

def tally():
    """
    System instruction that informs the child component
    to tally operations.
    """
    bhc.bhc_tally()

def get_data_pointer(ary, allocate=False, nullify=False):
    """Retrieves the data pointer from Bohrium Runtime."""
    if ary.size == 0 or ary.base.size == 0:
        return 0

    dtype = dtype_name(ary)
    ary = ary.bhc_obj
    exec("bhc.bhc_sync_A%s(ary)" % dtype)
    exec("bhc.bhc_discard_A%s(ary)" % dtype)
    exec("bhc.bhc_flush()")
    exec("data = bhc.bhc_data_get_A%s(ary, allocate, nullify)" % dtype)
    if data is None:
        if not allocate:
            return 0
        else:
            raise MemoryError()
    return int(data)

def set_bhc_data_from_ary(self, ary):
    """Assigns the data using memmove."""

    dtype = dtype_name(self)
    assert dtype == dtype_name(ary)
    ptr = get_data_pointer(self, allocate=True, nullify=False)
    ctypes.memmove(ptr, ary.ctypes.data, ary.dtype.itemsize * ary.size)

def ufunc(op, *args, **kwd):
    """
    Apply the 'op' on args, which is the output followed by one or two inputs
    Use the 'dtypes' option in 'kwd' to force the data types (None is default)

    :op npbackend.ufunc.Ufunc: Instance of a Ufunc.
    :args *?: Probably any one of ndarray, Base, Scalar, View, and npscalar.
    :rtype: None
    """

    dtypes = kwd.get("dtypes", [None]*len(args))

    if hasattr(op, "info"):#Make sure that 'op' is the operation name
        op = op.info['name']

    # The dtype of the scalar argument (if any) is the same as the array input
    scalar_type = None
    for arg in args[1:]:
        if not numpy.isscalar(arg):
            scalar_type = dtype_name(arg)
            break
    if scalar_type is None:#All inputs are scalars
        if len(args) == 1:
            scalar_type = dtype_name(args[0])
        else:
            scalar_type = dtype_name(args[1])

    fname  = "bhc.bhc_%s"%op
    for arg, dtype in zip(args, dtypes):
        if numpy.isscalar(arg):
            if dtype is None:
                fname += "_K%s"%scalar_type
            else:
                fname += "_K%s"%dtype_name(dtype)
        else:
            if dtype is None:
                fname += "_A%s"%dtype_name(arg)
            else:
                fname += "_A%s"%dtype_name(dtype)

    _bhc_exec(eval(fname), *args)

def matmul(out, in1, in2):
    """
    Perform matrix multiplication of 'in1' and 'in2' and store it in 'out'.

    :out ?:
    :in1 ?:
    :in2 ?:
    :rtype: None
    """
    ufunc("matmul", out, in1, in2)

def reduce(op, out, ary, axis):
    """
    reduce 'axis' dimension of 'ary' and write the result to out

    :op npbackend.ufunc.Ufunc: Instance of a Ufunc.
    """
    if ary.size == 0 or ary.base.size == 0:
        return

    ufunc("%s_reduce"%op.info['name'], out, ary, axis, dtypes=[None,None,numpy.dtype("int64")])


def accumulate(op, out, ary, axis):
    """
    Accumulate 'axis' dimension of 'ary' and write the result to out

    :op npbackend.ufunc.Ufunc: Instance of a Ufunc.
    :out ?:
    :in1 ?:
    :in2 ?:
    :rtype: None
    """
    if ary.size == 0 or ary.base.size == 0:
        return

    ufunc("%s_accumulate"%op.info['name'], out, ary, axis, dtypes=[None,None,numpy.dtype("int64")])

def extmethod(name, out, in1, in2):
    """
    Apply the extended method 'name'

    :name str: Name of the extension method.
    :out ?:
    :in1 ?:
    :in2 ?:
    :rtype: None
    """
    if out.size == 0 or out.base.size == 0:
        return
    func = eval("bhc.bhc_extmethod_A%s_A%s_A%s" % (
        dtype_name(out),
        dtype_name(in1),
        dtype_name(in2)
    ))
    ret = _bhc_exec(func, name, out, in1, in2)

    if ret != 0:
        raise NotImplementedError("The current runtime system does not support "
                                  "the extension method '%s'" % name)

def range(size, dtype):
    """
    Create a new array containing the values [0:size[

    :size int: Number of elements in the range [0:size[
    :in1 numpy.dtype: The
    :rtype: None
    """

    #Create new array
    ret = View(1, 0, (size,), (1,), Base(size, dtype))

    #And apply the range operation
    if size > 0:
        ufunc("range", ret)
    return ret

def random123(size, start_index, key):
    """
    Create a new random array using the random123 algorithm.
    The dtype is uint64 always.
    """

    dtype = numpy.dtype("uint64")

    #Create new array
    ret = View(1, 0, (size,), (1,), Base(size, dtype))

    #And apply the range operation
    if size > 0:
        ufunc("random123", ret, start_index, key, dtypes=[dtype]*3)
    return ret

def gather(out, ary, indexes):
    """
    Gather elements from 'ary' selected by 'indexes'.
    ary.shape == indexes.shape.

    :param Mixed out: The array to write results to.
    :param Mixed ary: Input array.
    :param Mixed indexes: Array of indexes (uint64).
    """

    ufunc("gather", out, ary, indexes)

