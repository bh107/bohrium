# -*- coding: utf-8 -*-
import operator
import functools
import math
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


class BhArray(object):
    def __init__(self, shape, dtype, stride=None, offset=0, base=None, is_scalar=False):
        if np.isscalar(shape):
            shape = (shape,)
        self.dtype = _dtype_util.type_to_dtype(dtype)
        self._bh_dtype_enum = _dtype_util.np2bh_enum(self.dtype)
        if is_scalar:
            assert (len(shape) == 0)
            self.nelem = 1
        elif len(shape) == 0:
            self.nelem = 0
        else:
            self.nelem = functools.reduce(operator.mul, shape)
        if base is None:
            base = BhBase(self.dtype, self.nelem)
        assert (self.dtype == base.dtype)
        if stride is None:
            stride = [0] * len(shape)
            s = 1
            for i in reversed(range(len(shape))):
                stride[i] = s
                s *= shape[i]
        self.base = base
        self.shape = tuple(shape)
        self.stride = tuple(stride)
        self.offset = offset
        if self.nelem == 0:
            self._bhc_handle = None
        else:
            if is_scalar:  # BhArray can be a scalar but the underlying bhc array is always an array
                shape = (1,)
                stride = (1,)

            self._bhc_handle = _bh_api.view(self._bh_dtype_enum, base._bhc_handle, len(shape),
                                            int(offset), list(shape), list(stride))

    @classmethod
    def from_scalar(cls, scalar):
        ret = cls(shape=(1.), dtype=_dtype_util.obj_to_dtype(scalar), is_scalar=True)
        ret.fill(scalar)
        return ret

    @classmethod
    def from_numpy(cls, numpy_array):
        numpy_array = np.require(numpy_array, requirements=['C_CONTIGUOUS', 'ALIGNED', 'OWNDATA'])
        ret = cls(numpy_array.shape, numpy_array.dtype,
                  stride=[s // numpy_array.itemsize for s in numpy_array.strides],
                  is_scalar=numpy_array.ndim == 0)
        _bh_api.copy_from_memory_view(ret._bh_dtype_enum, ret._bhc_handle, memoryview(numpy_array))
        return ret

    @classmethod
    def from_object(cls, obj):
        return cls.from_numpy(np.array(obj))

    def __del__(self):
        if hasattr(self, '_bhc_handle') and self._bhc_handle is not None:
            _bh_api.destroy(self.base._bh_dtype_enum, self._bhc_handle)

    def __str__(self):
        if self.nelem == 0:
            return "[]"
        else:
            return str(self.asnumpy())

    def view(self):
        return BhArray(self.shape, self.dtype, self.stride, self.offset, self.base,
                       is_scalar=self.nelem == 1 and len(self.shape) == 0)

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
        length = functools.reduce(operator.mul, shape)
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

    def __getitem_at_dim(self, dim, key):
        if np.isscalar(key):
            if not isinstance(key, _dtype_util.integers):
                raise IndexError("Only integers, slices (`:`), ellipsis (`...`), np.newaxis (`None`) and "
                                 "integer or boolean arrays are valid indices")
            if len(self.shape) <= dim or key >= self.shape[dim]:
                raise IndexError("Index out of bound")
            shape = list(self.shape)
            shape.pop(dim)
            stride = list(self.stride)
            stride.pop(dim)
            offset = self.offset + key * self.stride[dim]
            return BhArray(shape, self.dtype, offset=offset, stride=stride, base=self.base, is_scalar=len(shape) == 0)
        elif isinstance(key, slice):
            if len(self.shape) <= dim:
                raise IndexError("IndexError: too many indices for array")
            length = self.shape[dim]
            # complete missing slice information
            step = 1 if key.step is None else key.step
            if key.start is None:
                start = length - 1 if step < 0 else 0
            else:
                start = key.start
                if start < 0:
                    start += length
                if start < 0:
                    start = -1 if step < 0 else 0
                if start >= length:
                    start = length - 1 if step < 0 else length
            if key.stop is None:
                stop = -1 if step < 0 else length
            else:
                stop = key.stop
                if stop < 0:
                    stop += length
                if stop < 0:
                    stop = -1
                if stop > length:
                    stop = length
            new_length = int(math.ceil(abs(stop - start) / float(abs(step))))
            shape = list(self.shape[:dim]) + [new_length] + list(self.shape[dim + 1:])
            stride = list(self.stride[:dim]) + [step * self.stride[dim]] + list(self.stride[dim + 1:])
            offset = self.offset + start * self.stride[dim]
            return BhArray(shape, self.dtype, offset=offset, stride=stride, base=self.base)
        elif key is None:
            shape = list(self.shape)
            shape.insert(dim, 1)
            stride = list(self.stride)
            stride.insert(dim, 0)
            return BhArray(shape, self.dtype, offset=self.offset, stride=stride, base=self.base)
        else:
            raise IndexError("Only integers, slices (`:`), ellipsis (`...`), np.newaxis (`None`) and "
                             "integer or boolean arrays are valid indices")

    def __getitem__(self, key):
        if np.isscalar(key) or isinstance(key, slice) or key is None or key is Ellipsis:
            key = (key,)

        if isinstance(key, tuple):
            key = list(key)
            if Ellipsis in key:
                if key.count(Ellipsis) > 1:
                    raise IndexError("An index can only have a single ellipsis ('...')")

                # We inserts `slice(None, None, None)` at the position of the ellipsis
                # until `key` has the size of the number of dimension.
                idx = key.index(Ellipsis)
                while len(key) < len(self.shape) + 1:
                    key.insert(idx + 1, slice(None, None, None))
                key.pop(idx)
                assert (len(key) == len(self.shape))

            for i, k in enumerate(key):
                if k != slice(None, None, None):
                    ret = self.__getitem_at_dim(i, k)
                    key.pop(i)
                    if not np.isscalar(k):
                        key.insert(i, slice(None, None, None))
                    return ret.__getitem__(tuple(key))
            else:
                return self.view()

        raise IndexError("Only integers, slices (`:`), ellipsis (`...`), np.newaxis (`None`) and "
                         "integer or boolean arrays are valid indices")

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
