"""
Abstract interface for computation backend targets

"""

class Base(object):
    """
    Abstract base array handle (a array has only one base)
    Encapsulates memory allocated for an array.

    :size int: Number of elements in the array
    :dtype numpy.dtype: Data type of the elements
    """
    def __init__(self, size, dtype):
        self.size = size        # Total number of elements
        self.dtype = dtype      # Data type

class View(object):
    """
    Abstract array view handle.
    Encapsulates meta-data of an array.
    
    :ndim int: Number of dimensions / rank of the view
    :start int: Offset from base (in elements), converted to offset from base
                when constructed.
    :shape tuple(int*ndim): Number of elements in each dimension of the array.
    :strides tuple(int*ndim):Stride for each dimension (in elements), converted
    to stride for each dimension (in bytes) upon construction.
    :base interface.Base: Base of the 
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

    NOTE: One implementation could be to return ndarray.ctypes.data.
   
    :ary ?: ndarray? View? Base? Scalar?
    :allocate bool: When true the target is expected to allocate the data prior
    to returning.
    :nullify bool: What is this?
    :returns: A pointer to memory associated with the given 'ary'.
    :rtype: int
    """
    raise NotImplementedError()

def set_bhc_data_from_ary(self, ary):
    """Copy data from 'ary' into the array 'self'"""
    raise NotImplementedError()

def ufunc(op, *args):
    """Perform the ufunc 'op' on the 'args' arrays"""
    raise NotImplementedError()

def reduce(op, out, ary, axis):
    """Reduce 'axis' dimension of 'ary' and write the result to out"""
    raise NotImplementedError()

def accumulate(op, out, ary, axis):
    """Accumulate 'axis' dimension of 'ary' and write the result to out"""
    raise NotImplementedError()

def extmethod(name, out, in1, in2):
    """Apply the extended method 'name' """
    raise NotImplementedError()

def range(size, dtype):
    """Create a new array containing the values [0:size["""
    raise NotImplementedError()

def random123(size, start_index, key):
    """
    Create a new random array using the random123 algorithm.
    The dtype is uint64 always.
    """
    raise NotImplementedError()
