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
            func = eval("bhc.bhc_new_%s" % dtype_name(dtype))
            bhc_obj = _bhc_exec(func, size)
        self.bhc_obj = bhc_obj

    def __del__(self):
        if self.size == 0:
            return
        exec("bhc.bhc_destroy_%s(self.bhc_obj)" %
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
        func = eval("bhc.bhc_view_%s" % dtype)
        self.bhc_obj = func(base.bhc_obj, ndim, start, shape, strides)

    def __del__(self):
        if self.size == 0:
            return
        exec("bhc.bh_multi_array_%s_destroy(self.bhc_obj)" % dtype_name(
            self.dtype
        ))

def _bhc_exec(func, *args):
    """execute the 'func' with the bhc objects in 'args'"""

    args = list(args)
    for i in xrange(len(args)):
        if isinstance(args[i], View):
            args[i] = args[i].bhc_obj
    return func(*args)

def get_data_pointer(ary, allocate=False, nullify=False):
    """Retrieves the data pointer from Bohrium Runtime."""
    if ary.size == 0 or ary.base.size == 0:
        return 0

    dtype = dtype_name(ary)
    ary = ary.bhc_obj
    exec("bhc.bhc_multi_array_%s_sync(ary)" % dtype)
    exec("bhc.bh_multi_array_%s_discard(ary)" % dtype)
    exec("bhc.bhc_flush()")
    exec("data = bhc.bh_multi_array_%s_get_data(ary)" % dtype)
    if data is None:
        if not allocate:
            return 0
        exec("data = bhc.bh_multi_array_%s_get_data_and_force_alloc(ary)"
             % dtype
        )
        if data is None:
            raise MemoryError()
    if nullify:
        exec("bhc.bh_multi_array_%s_nullify_data(ary)"%dtype)
    return int(data)

def set_bhc_data_from_ary(self, ary):
    """Assigns the data using memmove."""

    dtype = dtype_name(self)
    assert dtype == dtype_name(ary)
    ptr = get_data_pointer(self, allocate=True, nullify=False)
    ctypes.memmove(ptr, ary.ctypes.data, ary.dtype.itemsize * ary.size)

def ufunc(op, *args):
    """
    Apply the 'op' on args, which is the output followed by one or two inputs

    :op npbackend.ufunc.Ufunc: Instance of a Ufunc.
    :args *?: Probably any one of ndarray, Base, Scalar, View, and npscalar.
    :rtype: None
    """

    scalar_str = ""
    in_dtype = dtype_name(args[1])
    for i, arg in enumerate(args):
        if numpy.isscalar(arg):
            if i == 1:
                scalar_str = "_scalar" + ("_lhs" if len(args) > 2 else "")
            if i == 2:
                scalar_str = "_scalar" + ("_rhs" if len(args) > 2 else "")
        elif i > 0:
            in_dtype = arg.dtype#overwrite with a non-scalar input
            #Do nothing on zero-sized arguments
            if arg.size == 0 or arg.base.size == 0:
                return

    if op.info['name'] == "identity":#Identity is a special case
        cmd = "bhc.bh_multi_array_%s_identity_%s" % (
            dtype_name(args[0].dtype),
            dtype_name(in_dtype)
        )
    else:
        cmd = "bhc.bh_multi_array_%s_%s" % (
            dtype_name(in_dtype), op.info['name']
        )
    cmd += scalar_str
    _bhc_exec(eval(cmd), *args)

def reduce(op, out, ary, axis):
    """
    reduce 'axis' dimension of 'ary' and write the result to out

    :op npbackend.ufunc.Ufunc: Instance of a Ufunc.
    """
    if ary.size == 0 or ary.base.size == 0:
        return

    func = eval("bhc.bh_multi_array_%s_%s_reduce" % (
        dtype_name(ary), op.info['name']
    ))
    _bhc_exec(func, out, ary, axis)

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

    func = eval("bhc.bh_multi_array_%s_%s_accumulate" % (
        dtype_name(ary), op.info['name'])
    )
    _bhc_exec(func, out, ary, axis)

def matmul(out, in1, in2):
    """
    Perform matrix multiplication of 'in1' and 'in2' and store it in 'out'.

    :out ?:
    :in1 ?:
    :in2 ?:
    :rtype: None
    """

    func = eval("bhc.bh_multi_array_%s_matmul" % dtype_name(out))
    _bhc_exec(func, out, in1, in2)

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

    func = eval("bhc.bh_multi_array_extmethod_%s_%s_%s" % (
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
    create a new array containing the values [0:size[

    :size int: Number of elements in the range [0:size[
    :in1 numpy.dtype: The
    :rtype: None
    """

    if size > 0:
        func = eval("bhc.bh_multi_array_%s_new_range"%dtype_name(dtype))
        bhc_obj = _bhc_exec(func, size)
    else:
        bhc_obj = None
    base = Base(size, dtype, bhc_obj)
    return View(1, 0, (size,), (dtype.itemsize,), base)

def random123(size, start_index, key):
    """
    Create a new random array using the random123 algorithm.
    The dtype is uint64 always.
    """

    dtype = numpy.dtype("uint64")
    if size > 0:
        func = eval("bhc.bh_multi_array_uint64_new_random123")
        bhc_obj = _bhc_exec(func, size, start_index, key)
    else:
        bhc_obj = None
    base = Base(size, dtype, bhc_obj)
    return View(1, 0, (size,), (dtype.itemsize,), base)


