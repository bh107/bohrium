"""
Bohrium C-backend as target for npbackend.
"""
import ctypes
import numpy
import functools
import operator
import atexit
import os
import sys

from .._util import dtype_name
from . import interface


class BhcAPI:
    """This class encapsulate the Bohrium C API
      NB: when Base.__del__() and View.__del__() is called, all other modules including 'bhc' and '_util' might already
      have been deallocated! Thus, initially we will manually load all 'bhc' functions in order to make this script
      completely self-contained.
    """

    def __init__(self):
        def get_bhc_api():
            bhc_py_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bhc.py")
            # We need to import '_bhc' manually into bhc_api since SWIG does know about the 'bohrium' package
            from .. import _bhc
            sys.modules['_bhc'] = _bhc
            try:
                bhc_api = {"__file__": bhc_py_path, "sys": sys}
                execfile(bhc_py_path, bhc_api)  # execfile updates 'bhc_api'
                return bhc_api
            except NameError:
                import runpy
                # run_path() returns the globals() from the run
                return runpy.run_path(bhc_py_path, init_globals={"sys": sys}, run_name="__main__")

        # Load the bhc API
        for key, val in get_bhc_api().items():
            if key.startswith("bhc_"):
                setattr(self, key[4:], val)  # Save key without the "bhc_"

    def __call__(self, name, *args, **kwargs):
        """Call the API function named `name` with the `*args` and `**kwargs`"""
        func = getattr(self, name)
        return func(*args, **kwargs)

    def call_single_dtype(self, name, dtype_name, *args, **kwargs):
        """Call the API function with only a single type signature it its name."""
        return self("%s_A%s" % (name, dtype_name), *args, **kwargs)


bhc = BhcAPI()


class Base(interface.Base):
    """ Base array handle """

    def __init__(self, size, dtype, bhc_obj=None):
        super(Base, self).__init__(size, dtype)
        if size == 0:
            return

        if bhc_obj is None:
            bhc_obj = bhc.call_single_dtype("new", self.dtype_name, size)

        self.bhc_obj = bhc_obj

    def __del__(self):
        if self.size == 0:
            return
        bhc.call_single_dtype("destroy", self.dtype_name, self.bhc_obj)


class View(interface.View):
    """ Array view handle """

    def __init__(self, ndim, start, shape, strides, base):
        super(View, self).__init__(ndim, start, shape, strides, base)
        self.size = functools.reduce(operator.mul, shape, 1)

        if self.size == 0:
            return
        self.bhc_obj = bhc.call_single_dtype("view", self.dtype_name, base.bhc_obj, ndim, start, shape, strides)

    def __del__(self):
        if self.size == 0:
            return
        bhc.call_single_dtype("destroy", self.dtype_name, self.bhc_obj)


def _bhc_exec(func, *args):
    """ Execute the 'func' with the bhc objects in 'args' """

    args = list(args)
    for i in range(len(args)):
        if isinstance(args[i], View):
            if not hasattr(args[i], 'bhc_obj'):
                # Ignore zero-sized views
                return
            args[i] = args[i].bhc_obj
    return func(*args)


def runtime_flush():
    """ Flush the runtime system """
    bhc.flush()


def runtime_flush_count():
    """Get the number of times flush has been called"""
    return bhc.flush_count()


def runtime_flush_and_repeat(nrepeats, ary):
    """Flush and repeat the lazy evaluated operations while `ary` is true and `nrepeats` hasn't been reach"""

    if ary is None:
        return bhc.flush_and_repeat(nrepeats)
    else:
        return bhc.flush_and_repeat_condition(nrepeats, ary.bhc_obj)


def runtime_sync(ary):
    """Sync `ary` to host memory"""

    dtype = dtype_name(ary)
    ary = ary.bhc_obj
    bhc.call_single_dtype("sync", dtype, ary)


def tally():
    pass


def get_data_pointer(ary, copy2host=True, allocate=False, nullify=False):
    """ Retrieves the data pointer from Bohrium Runtime. """
    if ary.size == 0 or ary.base.size == 0:
        return 0

    dtype = dtype_name(ary)
    ary = ary.bhc_obj

    if copy2host:
        bhc.call_single_dtype("sync", dtype, ary)
    runtime_flush()

    data = bhc.call_single_dtype("data_get", dtype, ary, copy2host, allocate, nullify)
    if data is None:
        if not allocate:
            return 0
        else:
            raise MemoryError()
    return int(data)


def set_data_pointer(ary, mem_ptr_as_int, host_ptr=True):
    """ Set the data pointer `mem_ptr_as_int` in the Bohrium Runtime. """
    if ary.size == 0 or ary.base.size == 0:
        return 0

    dtype = dtype_name(ary)
    ary = ary.bhc_obj

    bhc.call_single_dtype("sync", dtype, ary)
    runtime_flush()

    bhc.call_single_dtype("data_set", dtype, ary, host_ptr, mem_ptr_as_int)


def get_device_context():
    """Get the device context, such as OpenCL's cl_context, of the first VE in the runtime stack."""
    return int(bhc.get_device_context())


def set_device_context(device_context):
    """Set the device context, such as CUDA's cl_context, of the first VE in the runtime stack."""
    bhc.set_device_context(device_context)


def set_bhc_data_from_ary(self, ary):
    """ Assigns the data using memmove """

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

    dtypes = kwd.get("dtypes", [None] * len(args))

    # Make sure that 'op' is the operation name
    if hasattr(op, "info"):
        op = op.info['name']

    # The dtype of the scalar argument (if any) is the same as the array input
    scalar_type = None
    for arg in args[1:]:
        if not numpy.isscalar(arg):
            scalar_type = dtype_name(arg)
            break

    # All inputs are scalars
    if scalar_type is None:
        if len(args) == 1:
            scalar_type = dtype_name(args[0])
        else:
            scalar_type = dtype_name(args[1])

    fname = "%s" % op
    for arg, dtype in zip(args, dtypes):
        if numpy.isscalar(arg):
            if dtype is None:
                fname += "_K%s" % scalar_type
            else:
                fname += "_K%s" % dtype_name(dtype)
        else:
            if dtype is None:
                fname += "_A%s" % dtype_name(arg)
            else:
                fname += "_A%s" % dtype_name(dtype)

    _bhc_exec(getattr(bhc, fname), *args)


def reduce(op, out, ary, axis):
    """
    Reduce 'axis' dimension of 'ary' and write the result to out

    :op npbackend.ufunc.Ufunc: Instance of a Ufunc.
    """
    if ary.size == 0 or ary.base.size == 0:
        return

    ufunc("%s_reduce" % op.info['name'], out, ary, axis, dtypes=[None, None, numpy.dtype("int64")])


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

    ufunc("%s_accumulate" % op.info['name'], out, ary, axis, dtypes=[None, None, numpy.dtype("int64")])


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

    func = getattr(bhc, "extmethod_A%s_A%s_A%s" % (
        dtype_name(out),
        dtype_name(in1),
        dtype_name(in2)
    ))

    ret = _bhc_exec(func, name, out, in1, in2)

    if ret != 0:
        raise NotImplementedError("The current runtime system does not support "
                                  "the extension method '%s'" % name)


def arange(size, dtype):
    """
    Create a new array containing the values [0:size[

    :size int: Number of elements in the range [0:size[
    :in1 numpy.dtype: The
    :rtype: None
    """

    # Create new array
    ret = View(1, 0, (size,), (1,), Base(size, dtype))

    # And apply the range operation
    if size > 0:
        ufunc("range", ret)

    return ret


def random123(size, start_index, key):
    """
    Create a new random array using the random123 algorithm.
    The dtype is uint64 always.
    """

    dtype = numpy.dtype("uint64")

    # Create new array
    ret = View(1, 0, (size,), (1,), Base(size, dtype))

    # And apply the range operation
    if size > 0:
        ufunc("random123", ret, start_index, key, dtypes=[dtype] * 3)

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


def scatter(out, ary, indexes):
    """
    Scatter elements from 'ary' into 'out' at locations specified by 'indexes'.
    ary.shape == indexes.shape.

    :param Mixed out: The array to write results to.
    :param Mixed ary: Input array.
    :param Mixed indexes: Array of absolute indexes (uint64).
    """

    ufunc("scatter", out, ary, indexes)


def cond_scatter(out, ary, indexes, mask):
    """
    Scatter elements from 'ary' into 'out' at locations specified by 'indexes' where 'mask' is true.
    ary.shape == indexes.shape.

    :param Mixed out: The array to write results to.
    :param Mixed ary: Input array.
    :param Mixed indexes: Array of absolute indexes (uint64).
    :param Mixed ary: A boolean mask that specifies which indexes and values to include and exclude
    """

    ufunc("cond_scatter", out, ary, indexes, mask)


def message(msg):
    """ Send and receive a message through the component stack """
    return "%s" % (bhc.message(msg))


@atexit.register
def shutdown():
    """Actions to invoke when shutting down."""

    runtime_flush()
