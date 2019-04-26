# -*- coding: utf-8 -*-
import operator
import functools
import copy
import numpy as np
from bohrium_api import _bh_api, _info
from . import _dtype_util


class BhBase(object):
    def __init__(self, dtype, nelem):
        self.dtype = _dtype_util.any2np(dtype)
        self._bhc_handle = None
        self._bh_dtype_enum = _dtype_util.np2bh_enum(self.dtype)
        self.nelem = nelem
        self.itemsize = _dtype_util.size_of(self.dtype)
        self.nbytes = nelem * self.itemsize
        self._bhc_handle = _bh_api.new(self._bh_dtype_enum, self.nelem)

    def __del__(self):
        if self._bhc_handle is not None:
            _bh_api.destroy(self._bh_dtype_enum, self._bhc_handle)

    def __str__(self):
        return str(self.copy2numpy())

    def copy2numpy(self):
        _bh_api.flush()
        data = _bh_api.data_get(self._bh_dtype_enum, self._bhc_handle, True, True, False, self.nbytes)
        return np.frombuffer(data, dtype=self.dtype)


class BhArray(object):
    def __init__(self, shape, dtype, stride=None, offset=0, base=None):
        if np.isscalar(shape):
            shape = (shape,)
        self.dtype = _dtype_util.any2np(dtype)
        self._bhc_handle = None
        self._bh_dtype_enum = _dtype_util.np2bh_enum(self.dtype)
        self.nelem = functools.reduce(operator.mul, shape)
        if base is None:
            base = BhBase(self.dtype, self.nelem)
        assert (self.dtype == base.dtype)
        if stride is None:
            stride = [0] * len(shape)
            s = 1
            for i in range(len(shape)):
                stride[len(shape) - i - 1] = s
                s *= shape[i]
        self.base = base
        self.shape = tuple(shape)
        self.stride = tuple(stride)
        self.offset = offset
        self._bhc_handle = _bh_api.view(self._bh_dtype_enum, base._bhc_handle, len(shape),
                                        int(offset), list(shape), list(stride))

    def __del__(self):
        if self._bhc_handle is not None:
            _bh_api.destroy(self.base._bh_dtype_enum, self._bhc_handle)

    def __str__(self):
        return str(self.copy2numpy())

    def view(self):
        return copy.deepcopy(self)

    def copy2numpy(self):
        data = self.base.copy2numpy()
        if self.offset > 0:
            data = data[self.offset:]
        return np.lib.stride_tricks.as_strided(data, shape=self.shape,
                                               strides=[s * self.base.itemsize for s in self.stride])

    def fill(self, value):
        """Fill the array with a scalar value.

            Parameters
            ----------
            value : scalar
                All elements of `a` will be assigned this value.

            Examples
            --------
            >>> a = bh.array([1, 2])
            >>> a.fill(0)
            >>> a
            array([0, 0])
            >>> a = bh.empty(2)
            >>> a.fill(1)
            >>> a
            array([ 1.,  1.])
            """
        from .ufuncs import bhop_dict
        bhop_dict['identity'](value, self)

    def astype(self, dtype, copy=True):
        from .ufuncs import bhop_dict
        if not copy and self.dtype == dtype:
            return self
        ret = BhArray(self.shape, dtype, stride=self.stride, offset=self.offset, base=self.base)
        bhop_dict['identity'](self, ret)

