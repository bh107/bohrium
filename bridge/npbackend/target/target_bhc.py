"""
Bohrium C-backend as target for npbackend.
"""
import ctypes
import numpy
from .. import bhc
from .._util import dtype_name
from . import interface

class Base(interface.Base):
    """base array handle"""

    def __init__(self, size, dtype, bhc_obj=None):
        super(Base, self).__init__(size, dtype)

        if bhc_obj is None:
            func = eval("bhc.bh_multi_array_%s_new_empty" % dtype_name(dtype))
            bhc_obj = bhc_exec(func, 1, (size,))
        self.bhc_obj = bhc_obj

    def __del__(self):
        exec("bhc.bh_multi_array_%s_destroy(self.bhc_obj)" % 
             dtype_name(self.dtype)
        )

class View(interface.View):
    """array view handle"""

    def __init__(self, ndim, start, shape, strides, base):
        super(View, self).__init__(ndim, start, shape, strides, base)
        dtype = dtype_name(self.dtype)
        exec("base = bhc.bh_multi_array_%s_get_base(base.bhc_obj)"%dtype)
        func = eval("bhc.bh_multi_array_%s_new_from_view" % dtype)
        self.bhc_obj = func(base, ndim, start, shape, strides)

    def __del__(self):
        exec("bhc.bh_multi_array_%s_destroy(self.bhc_obj)" % dtype_name(
            self.dtype
        ))

def views2bhc(views):
    """returns the bhc objects in the 'views' but don't touch scalars"""

    singleton = not (hasattr(views, "__iter__") or
                     hasattr(views, "__getitem__"))
    if singleton:
        views = (views,)
    ret = []
    for view in views:
        if not numpy.isscalar(view):
            view = view.bhc_obj
        ret.append(view)
    if singleton:
        ret = ret[0]
    return ret

def bhc_exec(func, *args):
    """execute the 'func' with the bhc objects in 'args'"""

    args = list(args)
    for i in xrange(len(args)):
        if isinstance(args[i], View):
            args[i] = views2bhc(args[i])
    return func(*args)

def get_data_pointer(ary, allocate=False, nullify=False):
    """Retrieves the data pointer from Bohrium Runtime."""

    dtype = dtype_name(ary)
    ary = views2bhc(ary)
    exec("bhc.bh_multi_array_%s_sync(ary)" % dtype)
    exec("bhc.bh_multi_array_%s_discard(ary)" % dtype)
    exec("bhc.bh_runtime_flush()")
    exec("base = bhc.bh_multi_array_%s_get_base(ary)" % dtype)
    exec("data = bhc.bh_multi_array_%s_get_base_data(base)" % dtype)
    if data is None:
        if not allocate:
            return 0
        exec("data = bhc.bh_multi_array_%s_get_base_data_and_force_alloc(base)"
             % dtype
        )
        if data is None:
            raise MemoryError()
    if nullify:
        exec("bhc.bh_multi_array_%s_nullify_base_data(base)"%dtype)
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
    bhc_exec(eval(cmd), *args)

def reduce(op, out, ary, axis):
    """
    reduce 'axis' dimension of 'ary' and write the result to out
    
    :op npbackend.ufunc.Ufunc: Instance of a Ufunc.
    """

    func = eval("bhc.bh_multi_array_%s_%s_reduce" % (
        dtype_name(ary), op.info['name']
    ))
    bhc_exec(func, out, ary, axis)

def accumulate(op, out, ary, axis):
    """
    Accumulate 'axis' dimension of 'ary' and write the result to out
    
    :op npbackend.ufunc.Ufunc: Instance of a Ufunc.
    :out ?:
    :in1 ?:
    :in2 ?:
    :rtype: None
    """

    func = eval("bhc.bh_multi_array_%s_%s_accumulate" % (
        dtype_name(ary), op.info['name'])
    )
    bhc_exec(func, out, ary, axis)

def extmethod(name, out, in1, in2):
    """
    
    Apply the extended method 'name'
    
    :name str: Name of the extension method.
    :out ?:
    :in1 ?:
    :in2 ?:
    :rtype: None
    """

    func = eval("bhc.bh_multi_array_extmethod_%s_%s_%s" % (
        dtype_name(out),
        dtype_name(in1),
        dtype_name(in2)
    ))
    ret = bhc_exec(func, name, out, in1, in2)

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

    func = eval("bhc.bh_multi_array_%s_new_range" % dtype_name(dtype))
    bhc_obj = bhc_exec(func, size)
    base = Base(size, dtype, bhc_obj)

    return View(1, 0, (size,), (dtype.itemsize,), base)

def random123(size, start_index, key):
    """
    Create a new random array using the random123 algorithm.
    The dtype is uint64 always.
    """

    dtype = numpy.dtype("uint64")
    func = eval("bhc.bh_multi_array_uint64_new_random123")
    bhc_obj = bhc_exec(func, size, start_index, key)
    base = Base(size, dtype, bhc_obj)

    return View(1, 0, (size,), (dtype.itemsize,), base)

