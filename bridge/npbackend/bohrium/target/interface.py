"""
===================================
Interface for ``npbackend`` targets
===================================

Implementing a ``npbackend`` target, starts with fleshing out::

    import interface

    class View(interface.View):
        ...

    class Base(interface.Base):
        ...

And then implementing each of the methods described in ``interface.py``,
documented below:
"""
from .._util import dtype_name


class Base(object):
    """
    Abstract base array handle (an array has only one base)
    Encapsulates memory allocated for an array.

    :param int size: Number of elements in the array
    :param numpy.dtype dtype: Data type of the elements
    """

    def __init__(self, size, dtype):
        # Total number of elements
        self.size = size
        # Data type
        self.dtype = dtype
        # Data type name
        self.dtype_name = dtype_name(dtype)


class View(object):
    """
    Abstract array view handle.
    Encapsulates meta-data of an array.

    :param int ndim: Number of dimensions / rank of the view
    :param int start: Offset from base (in elements), converted to bytes upon construction.
    :param tuple(int*ndim) shape: Number of elements in each dimension of the array.
    :param tuple(int*ndim) strides: Stride for each dimension (in elements), converted to bytes upon construction.
    :param interface.Base base: Base associated with array.
    """

    def __init__(self, ndim, start, shape, strides, base):
        # Number of dimensions
        self.ndim = ndim
        # Tuple of dimension sizes
        self.shape = shape
        # The base array this view refers to
        self.base = base
        # Data type name
        self.dtype_name = base.dtype_name
        # Data type
        self.dtype = base.dtype
        # Offset from base (in bytes)
        self.start = start * base.dtype.itemsize
        # Tuple of strides (in bytes)
        self.strides = [x * base.dtype.itemsize for x in strides]


def runtime_flush():
    """ Flush the runtime system """
    pass


def runtime_flush_count():
    """Get the number of times flush has been called"""
    pass


def runtime_flush_and_repeat(nrepeats, ary):
    """Flush and repeat the lazy evaluated operations while `ary` is true and `nrepeats` hasn't been reach"""
    pass


def runtime_sync(ary):
    """Sync `ary` to host memory"""
    pass


def tally():
    """ Tally the runtime system """
    pass


def get_data_pointer(ary, copy2host=True, allocate=False, nullify=False):
    """
    Return a C-pointer to the array data (represented as a Python integer).

    .. note:: One way of implementing this would be to return a ndarray.ctypes.data.

    :param Mixed ary: The array to retrieve a data-pointer for.
    :param bool copy2host: When true always copy the memory to main memory.
    :param bool allocate: When true the target is expected to allocate the data prior to returning.
    :param bool nullify: TODO
    :returns: A pointer to memory associated with the given 'ary'
    :rtype: int
    """
    raise NotImplementedError()


def set_data_pointer(ary, mem_ptr_as_int, host_ptr=True):
    """ Set the data pointer `mem_ptr_as_int` in the Bohrium Runtime. """
    raise NotImplementedError()


def get_device_context():
    """Get the device context, such as OpenCL's cl_context, of the first VE in the runtime stack."""
    raise NotImplementedError()


def set_device_context(device_context):
    """Set the device context, such as CUDA's cl_context, of the first VE in the runtime stack."""
    raise NotImplementedError()


def set_bhc_data_from_ary(self, ary):
    """
    Copy data from 'ary' into the array 'self'

    :param Mixed self: The array to copy data to.
    :param Mixed ary: The array to copy data from.
    :rtype: None
    """
    raise NotImplementedError()


def ufunc(op, *args):
    """
    Perform the ufunc 'op' on the 'args' arrays

    :param bohrium.ufunc.Ufunc op: The ufunc operation to apply to args.
    :param Mixed args: Args to the ufunc operation.
    :rtype: None
    """
    raise NotImplementedError()


def reduce(op, out, ary, axis):
    """
    Reduce 'axis' dimension of 'ary' and write the result to out

    :param op bohrium.ufunc.Ufunc: The ufunc operation to apply to args.
    :param out Mixed: The array to reduce "into".
    :param ary Mixed: The array to reduce.
    :param axis Mixed: The axis to apply the reduction over.
    :rtype: None
    """
    raise NotImplementedError()


def accumulate(op, out, ary, axis):
    """
    Accumulate/scan 'axis' dimension of 'ary' and write the result to 'out'.

    :param bohrium.ufunc.Ufunc op: The element-wise operator to accumulate.
    :param Mixed out: The array to accumulate/scan "into".
    :param Mixed ary: The array to accumulate/scan.
    :param Mixed axis: The axis to apply the accumulation/scan over.
    :rtype: None
    """
    raise NotImplementedError()


def extmethod(name, out, in1, in2):
    """
    Apply the extension method 'name'.

    :param Mixed out: The array to write results to.
    :param Mixed in1: First input array.
    :param Mixed in2: Second input array.
    :rtype: None
    """
    raise NotImplementedError()


def arange(size, dtype):
    """
    Create a new array containing the values [0:size[.

    :param int size: Number of elements in the returned array.
    :param np.dtype dtype: Type of elements in the returned range.
    :rtype: Mixed
    """
    raise NotImplementedError()


def random123(size, start_index, key):
    """
    Create a new random array using the random123 algorithm.
    The dtype is uint64 always.

    :param int size: Number of elements in the returned array.
    :param int start_index: TODO
    :param int key: TODO
    """
    raise NotImplementedError()


def gather(out, ary, indexes):
    """
    Gather elements from 'ary' selected by 'indexes'.
    out.shape == indexes.shape.

    :param Mixed out: The array to write results to.
    :param Mixed ary: Input array.
    :param Mixed indexes: Array of absolute indexes (uint64).
    """
    raise NotImplementedError()


def scatter(out, ary, indexes):
    """
    Scatter elements from 'ary' into 'out' at locations specified by 'indexes'.
    ary.shape == indexes.shape.

    :param Mixed out: The array to write results to.
    :param Mixed ary: Input array.
    :param Mixed indexes: Array of absolute indexes (uint64).
    """
    raise NotImplementedError()


def cond_scatter(out, ary, indexes, mask):
    """
    Scatter elements from 'ary' into 'out' at locations specified by 'indexes' where 'mask' is true.
    ary.shape == indexes.shape.

    :param Mixed out: The array to write results to.
    :param Mixed ary: Input array.
    :param Mixed indexes: Array of absolute indexes (uint64).
    :param Mixed ary: A boolean mask that specifies which indexes and values to include and exclude
    """
    raise NotImplementedError()


def message(msg):
    """ Send and receive a message through the component stack """
    raise NotImplementedError()
