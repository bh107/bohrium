"""
The Computation Backend

"""
import ctypes
import numpy
from .. import bhc
from .._util import dtype_name
import backend

class base(backend.base):
    """base array handle"""
    def __init__(self, size, dtype, bhc_obj=None):
        super(base, self).__init__(size, dtype)
        if bhc_obj is None:
            f = eval("bhc.bh_multi_array_%s_new_empty"%dtype_name(dtype))
            bhc_obj = bhc_exec(f, 1, (size,))
        self.bhc_obj = bhc_obj

    def __del__(self):
        exec "bhc.bh_multi_array_%s_destroy(self.bhc_obj)"%dtype_name(self.dtype)

class view(backend.view):
    """array view handle"""
    def __init__(self, ndim, start, shape, stride, base):
        super(view, self).__init__(ndim, start, shape, stride, base)
        dtype = dtype_name(self.dtype)
        exec "base = bhc.bh_multi_array_%s_get_base(base.bhc_obj)"%dtype
        f = eval("bhc.bh_multi_array_%s_new_from_view"%dtype)
        self.bhc_obj = f(base, ndim, start, shape, stride)

    def __del__(self):
        exec "bhc.bh_multi_array_%s_destroy(self.bhc_obj)"%dtype_name(self.dtype)

def views2bhc(views):
    """returns the bhc objects in the 'views' but don't touch scalars"""
    singleton = not (hasattr(views, "__iter__") or
                     hasattr(views, "__getitem__"))
    if singleton:
        views = (views,)
    ret = []
    for v in views:
        if not numpy.isscalar(v):
            v = v.bhc_obj
        ret.append(v)
    if singleton:
        ret = ret[0]
    return ret

def bhc_exec(func, *args):
    """execute the 'func' with the bhc objects in 'args'"""
    args = list(args)
    for i in xrange(len(args)):
        if isinstance(args[i], view):
            args[i] = views2bhc(args[i])
    return func(*args)

def get_data_pointer(ary, allocate=False, nullify=False):
    dtype = dtype_name(ary)
    ary = views2bhc(ary)
    exec "bhc.bh_multi_array_%s_sync(ary)"%dtype
    exec "bhc.bh_multi_array_%s_discard(ary)"%dtype
    exec "bhc.bh_runtime_flush()"
    exec "base = bhc.bh_multi_array_%s_get_base(ary)"%dtype
    exec "data = bhc.bh_multi_array_%s_get_base_data(base)"%dtype
    if data is None:
        if not allocate:
            return 0
        exec "data = bhc.bh_multi_array_%s_get_base_data_and_force_alloc(base)"%dtype
        if data is None:
            raise MemoryError()
    if nullify:
        exec "bhc.bh_multi_array_%s_nullify_base_data(base)"%dtype
    return int(data)

def set_bhc_data_from_ary(self, ary):
    dtype = dtype_name(self)
    assert dtype == dtype_name(ary)
    d = get_data_pointer(self, allocate=True, nullify=False)
    ctypes.memmove(d, ary.ctypes.data, ary.dtype.itemsize * ary.size)

def ufunc(op, *args):
    """Apply the 'op' on args, which is the output followed by one or two inputs"""
    scalar_str = ""
    in_dtype = dtype_name(args[1])
    for i,a in enumerate(args):
        if numpy.isscalar(a):
            if i == 1:
                scalar_str = "_scalar" + ("_lhs" if len(args) > 2 else "")
            if i == 2:
                scalar_str = "_scalar" + ("_rhs" if len(args) > 2 else "")
        elif i > 0:
            in_dtype = a.dtype#overwrite with a non-scalar input

    if op.info['name'] == "identity":#Identity is a special case
        cmd = "bhc.bh_multi_array_%s_identity_%s"%(dtype_name(args[0].dtype), dtype_name(in_dtype))
    else:
        cmd = "bhc.bh_multi_array_%s_%s"%(dtype_name(in_dtype), op.info['name'])
    cmd += scalar_str
    bhc_exec(eval(cmd), *args)

def reduce(op, out, a, axis):
    """reduce 'axis' dimension of 'a' and write the result to out"""

    f = eval("bhc.bh_multi_array_%s_%s_reduce"%(dtype_name(a), op.info['name']))
    bhc_exec(f, out, a, axis)

def accumulate(op, out, a, axis):
    """accumulate 'axis' dimension of 'a' and write the result to out"""

    f = eval("bhc.bh_multi_array_%s_%s_accumulate"%(dtype_name(a), op.info['name']))
    bhc_exec(f, out, a, axis)

def extmethod(name, out, in1, in2):
    """Apply the extended method 'name' """

    f = eval("bhc.bh_multi_array_extmethod_%s_%s_%s"%(dtype_name(out),\
              dtype_name(in1), dtype_name(in2)))
    ret = bhc_exec(f, name, out, in1, in2)
    if ret != 0:
        raise NotImplementedError("The current runtime system does not support "
                                  "the extension method '%s'"%name)

def range(size, dtype):
    """create a new array containing the values [0:size["""

    f = eval("bhc.bh_multi_array_%s_new_range"%dtype_name(dtype))
    bhc_obj = bhc_exec(f, size)
    b = base(size, dtype, bhc_obj)
    return view(1, 0, (size,), (dtype.itemsize,), b)

def random123(size, start_index, key):
    """Create a new random array using the random123 algorithm.
    The dtype is uint64 always."""

    dtype = numpy.dtype("uint64")
    f = eval("bhc.bh_multi_array_uint64_new_random123")
    bhc_obj = bhc_exec(f, size, start_index, key)
    b = base(size, dtype, bhc_obj)
    return view(1, 0, (size,), (dtype.itemsize,), b)

