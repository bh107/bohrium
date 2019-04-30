# -*- coding: utf-8 -*-
import operator
import functools
import copy
import numpy as np
from bohrium_api import _bh_api
from . import _dtype_util


class BhBase(object):
    def __init__(self, dtype, nelem):
        self.dtype = _dtype_util.type_to_dtype(dtype)
        self._bh_dtype_enum = _dtype_util.np2bh_enum(self.dtype)
        self.nelem = nelem
        self.itemsize = _dtype_util.size_of(self.dtype)
        self.nbytes = nelem * self.itemsize
        if self.nelem == 0:
            self._bhc_handle = None
        else:
            self._bhc_handle = _bh_api.new(self._bh_dtype_enum, self.nelem)

    def __del__(self):
        if hasattr(self, '_bhc_handle') and self._bhc_handle is not None:
            _bh_api.destroy(self._bh_dtype_enum, self._bhc_handle)

    def __str__(self):
        return str(self.copy2numpy())


class BhArray(object):
    def __init__(self, shape, dtype, stride=None, offset=0, base=None):
        if np.isscalar(shape):
            shape = (shape,)
        self.dtype = _dtype_util.type_to_dtype(dtype)
        self._bh_dtype_enum = _dtype_util.np2bh_enum(self.dtype)
        if len(shape) == 0:
            self.nelem = 0
        else:
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
        if self.nelem == 0:
            self._bhc_handle = None
        else:
            self._bhc_handle = _bh_api.view(self._bh_dtype_enum, base._bhc_handle, len(shape),
                                            int(offset), list(shape), list(stride))

    @classmethod
    def fromNumpy(cls, numpy_array):
        numpy_array = np.require(numpy_array, requirements=['C_CONTIGUOUS', 'ALIGNED', 'OWNDATA'])
        ret = cls(numpy_array.shape, numpy_array.dtype, stride=[s // numpy_array.itemsize for s in numpy_array.strides])
        _bh_api.copy_from_memory_view(ret._bh_dtype_enum, ret._bhc_handle, memoryview(numpy_array))
        return ret

    def __del__(self):
        if hasattr(self, '_bhc_handle') and self._bhc_handle is not None:
            _bh_api.destroy(self.base._bh_dtype_enum, self._bhc_handle)

    def __str__(self):
        return str(self.asnumpy())

    def view(self):
        return copy.deepcopy(self)

    def asnumpy(self, flush=True):
        if self.nelem == 0:
            raise RuntimeError("The size of the zero!")
        if flush:
            _bh_api.flush()
        data = _bh_api.data_get(self._bh_dtype_enum, self._bhc_handle, True, True, False, self.base.nbytes)
        ret = np.frombuffer(data, dtype=self.dtype, offset=self.offset * self.base.itemsize)
        return np.lib.stride_tricks.as_strided(ret, self.shape, [s * self.base.itemsize for s in self.stride])

    def copy2numpy(self, flush=True):
        if self.nelem == 0:
            return np.array([], self.dtype)
        else:
            return self.asnumpy(flush).copy()

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
        from .ufuncs import assign
        assign(value, self)

    def astype(self, dtype, always_copy=True):
        from .ufuncs import assign
        if not always_copy and self.dtype == dtype:
            return self
        ret = BhArray(self.shape, dtype)
        assign(self, ret)
        return ret

    def iscontiguous(self):
        acc = 1
        for shape, stride in zip(self.shape, self.stride):
            if shape > 1 and stride != acc:
                return False
            else:
                acc *= shape
        return True

    def reshape(self, shape):
        from .ufuncs import assign
        length = self.nelem = functools.reduce(operator.mul, shape)
        if length != self.nelem:
            raise RuntimeError("Total size cannot change when reshaping")

        if self.iscontiguous():
            return BhArray(shape, self.dtype, offset=self.offset, base=self.base)
        else:
            ret = BhArray(shape, self.dtype)
            assign(self, ret)
            return ret

    def copy(self):
        """Return a copy of the array.

        Returns
        -------
        out : BhArray
            Copy of `self`
        """
        return self.astype(self.dtype, always_copy=True)

    # Binary Operators
    def __add__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['add'](self, other)

    def __iadd__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['add'](self, other, self)

    def __sub__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['subtract'](self, other)

    def __isub__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['subtract'](self, other, self)

    def __mul__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['multiply'](self, other)

    def __imul__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['multiply'](self, other, self)

    def __floordiv__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['floor_divide'](self, other)

    def __ifloordiv__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['floor_divide'](self, other, self)

    def __truediv__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['true_divide'](self, other)

    def __div__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['true_divide'](self, other)

    def __idiv__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['true_divide'](self, other, self)

    def __mod__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['mod'](self, other)

    def __imod__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['mod'](self, other, self)

    def __pow__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['power'](self, other)

    def __ipow__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['power'](self, other, self)

    def __and__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['bitwise_and'](self, other)

    def __iand__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['bitwise_and'](self, other, self)

    def __xor__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['bitwise_xor'](self, other)

    def __ixor__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['bitwise_xor'](self, other, self)

    def __or__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['bitwise_or'](self, other)

    def __ior__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['bitwise_or'](self, other, self)

    # Unary Operators
    def __neg__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['negative'](self, other)

    def __abs__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['absolute'](self, other)

    def __invert__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['invert'](self, other)
