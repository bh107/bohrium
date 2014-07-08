"""Abstract module for computation backends"""

class base(object):
    """abstract base array handle"""
    def __init__(self, size, dtype):
        self.size = size
        self.dtype = dtype

class view(object):
    """abstract array view handle"""
    def __init__(self, ndim, start, shape, stride, base):
        self.ndim = ndim
        self.shape = shape
        self.base = base
        self.dtype = base.dtype
        self.start = start*self.dtype.itemsize
        self.stride = [x * self.dtype.itemsize for x in stride]
