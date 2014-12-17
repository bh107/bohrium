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

class Base(object):
    """
    Abstract base array handle (a array has only one base)
    Encapsulates memory allocated for an array.

    :param int size: Number of elements in the array
    :param numpy.dtype dtype: Data type of the elements
    """
    def __init__(self, size, dtype):
        self.size = size        # Total number of elements
        self.dtype = dtype      # Data type

class View(object):
    """
    Abstract array view handle.
    Encapsulates meta-data of an array.
    
    :param int ndim: Number of dimensions / rank of the view
    :param int start: Offset from base (in elements), converted to offset from base
                when constructed.
    :param tuple(int*ndim) shape: Number of elements in each dimension of the array.
    :param tuple(int*ndim) strides: Stride for each dimension (in elements), converted to stride for each dimension (in bytes) upon construction.
    :param interface.Base base: Base associated with array.
    """
    def __init__(self, ndim, start, shape, strides, base):
        self.ndim = ndim        # Number of dimensions
        self.shape = shape      # Tuple of dimension sizes
        self.base = base        # The base array this view refers to
        self.dtype = base.dtype
        self.start = start * base.dtype.itemsize # Offset from base (in bytes)
        self.strides = [x * base.dtype.itemsize for x in strides] #Tuple of strides (in bytes)

def get_data_pointer(ary, allocate=False, nullify=False):
    """
    Return a C-pointer to the array data (represented as a Python integer).

    .. note:: One way of implementing this would be to return a ndarray.ctypes.data.
   
    :param Mixed ary: The array to retrieve a data-pointer for.
    :param bool allocate: When true the target is expected to allocate the data prior to returning.
    :param bool nullify: TODO
    :returns: A pointer to memory associated with the given 'ary'
    :rtype: int
    """
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
    
    :param npbackend.ufunc.Ufunc op: The ufunc operation to apply to args.
    :param Mixed args: Args to the ufunc operation.
    :rtype: None
    """
    raise NotImplementedError()

def reduce(op, out, ary, axis):
    """
    Reduce 'axis' dimension of 'ary' and write the result to out
    
    :param op npbackend.ufunc.Ufunc: The ufunc operation to apply to args.
    :param out Mixed: The array to reduce "into".
    :param ary Mixed: The array to reduce.
    :param axis Mixed: The axis to apply the reduction over.
    :rtype: None
    """
    raise NotImplementedError()

def accumulate(op, out, ary, axis):
    """
    Accumulate/scan 'axis' dimension of 'ary' and write the result to 'out'.

    :param npbackend.ufunc.Ufunc op: The element-wise operator to accumulate.
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

def range(size, dtype):
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
