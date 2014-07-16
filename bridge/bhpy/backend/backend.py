"""Abstract module for computation backends"""

class base(object):
    """Abstract base array handle (a array has only one base)"""
    def __init__(self, size, dtype):
        self.size = size #Total number of elements
        self.dtype = dtype #Data type

class view(object):
    """Abstract array view handle"""
    def __init__(self, ndim, start, shape, stride, base):
        self.ndim = ndim #Number of dimensions
        self.shape = shape #Tuple of dimension sizes
        self.base = base #The base array this view refers to
        self.dtype = base.dtype
        self.start = start*base.dtype.itemsize #Offset from base (in bytes)
        self.stride = [x * base.dtype.itemsize for x in stride] #Tuple of strides (in bytes)

def get_data_pointer(ary, allocate=False, nullify=False):
    """Return a C-pointer to the array data (as a Python integer)"""
    raise NotImplementedError()

def set_bhc_data_from_ary(self, ary):
    """Copy data from 'ary' into the array 'self'"""
    raise NotImplementedError()

def ufunc(op, *args):
    """Perform the ufunc 'op' on the 'args' arrays"""
    raise NotImplementedError()

def reduce(op, out, a, axis):
    """Reduce 'axis' dimension of 'a' and write the result to out"""
    raise NotImplementedError()

def accumulate(op, out, a, axis):
    """Accumulate 'axis' dimension of 'a' and write the result to out"""
    raise NotImplementedError()

def extmethod(name, out, in1, in2):
    """Apply the extended method 'name' """
    raise NotImplementedError()

def range(size, dtype):
    """Create a new array containing the values [0:size["""
    raise NotImplementedError()

def random123(size, start_index, key):
    """Create a new random array using the random123 algorithm.
    The dtype is uint64 always."""
    raise NotImplementedError()
