# -*- coding: utf-8 -*-
import math
import numpy as np
# noinspection PyProtectedMember,PyUnresolvedReferences
from bohrium_api import _bh_api
from . import _dtype_util, util


class BhBase(object):
    """A base array that represent a block of memory.
    A base array is always the sole owner of a complete memory allocation.
    """

    def __init__(self, dtype, nelem):
        #: The data type of the base array
        self.dtype = _dtype_util.type_to_dtype(dtype)
        #: The backend enum that corresponds to `self.dtype`
        self._bh_dtype_enum = _dtype_util.np2bh_enum(self.dtype)
        #: Number of elements
        self.nelem = nelem
        #: Size of an element in bytes
        self.itemsize = _dtype_util.size_of(self.dtype)
        #: Total size of the base array in bytes
        self.nbytes = nelem * self.itemsize
        if self.nelem == 0:
            self._bhc_handle = None
        else:
            self._bhc_handle = _bh_api.new(self._bh_dtype_enum, self.nelem)

    def __del__(self):
        if hasattr(self, '_bhc_handle') and self._bhc_handle is not None:
            _bh_api.destroy(self._bh_dtype_enum, self._bhc_handle)


class BhArray(object):
    """A array that represent a *view* of a base array. Multiple array views can point to the same base array."""

    def __init__(self, shape, dtype, strides=None, offset=0, base=None, is_scalar=False):
        if np.isscalar(shape):
            shape = (shape,)
        dtype = _dtype_util.type_to_dtype(dtype)
        if strides is None:
            strides = util.get_contiguous_strides(shape)
        if is_scalar:
            assert (len(shape) == 0)
            self.nelem = 1
        else:
            self.nelem = util.total_size(shape)
        #: The base array
        self.base = BhBase(dtype, self.nelem) if base is None else base
        if self.dtype != self.base.dtype:
            raise ValueError("dtype must be identical to base.dtype (%s)" % self.base.dtype)
        self._shape = tuple(shape)
        # NB: `_strides` is in elements and not in bytes, which is different from NumPy.
        self._strides = tuple(strides)
        self.offset = offset
        if self.nelem == 0:
            self._bhc_handle = None
        else:
            if is_scalar:  # BhArray can be a scalar but the underlying bhc array is always an array
                shape = (1,)
                strides = (1,)
            self._bhc_handle = _bh_api.view(self.base._bh_dtype_enum, self.base._bhc_handle, len(shape),
                                            int(offset), list(shape), list(strides))

    @classmethod
    def from_scalar(cls, scalar):
        ret = cls(shape=(1,), dtype=_dtype_util.obj_to_dtype(scalar), is_scalar=True)
        ret.fill(scalar)
        return ret

    @classmethod
    def from_numpy(cls, numpy_array):
        numpy_array = np.require(numpy_array, requirements=['C_CONTIGUOUS', 'ALIGNED', 'OWNDATA'])
        ret = cls(numpy_array.shape, numpy_array.dtype,
                  strides=[s // numpy_array.itemsize for s in numpy_array.strides],
                  is_scalar=numpy_array.ndim == 0)
        _bh_api.copy_from_memory_view(ret.base._bh_dtype_enum, ret._bhc_handle, memoryview(numpy_array))
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

    @property
    def dtype(self):
        return self.base.dtype

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def shape(self):
        return tuple(self._shape)

    @shape.setter
    def shape(self, shape):
        if self.isscalar():
            raise ValueError("Cannot reshape a scalar")
        if util.total_size(shape) != util.total_size(self.shape):
            raise ValueError("Cannot reshape array of size %d into shape %s" % (util.total_size(self.shape), shape))
        if not self.iscontiguous():
            raise ValueError("Cannot reshape a non-contiguous array")
        self._shape = tuple(shape)
        self._strides = util.get_contiguous_strides(shape)

    @property
    def strides_in_bytes(self):
        """Gets the strides in bytes"""
        return tuple([s * self.base.itemsize for s in self._strides])

    @property
    def strides(self):
        """Gets the strides in elements"""
        return tuple(self._strides)

    @strides.setter
    def strides(self, strides):
        """Sets the strides in elements"""
        if self.isscalar():
            raise ValueError("Scalars does not have `strides`")
        if len(strides) != len(self.shape):
            raise ValueError("Strides must be same length as shape (%d)" % len(self.shape))
        self._strides = tuple(strides)

    def view(self):
        return BhArray(self._shape, self.dtype, self._strides, self.offset, self.base,
                       is_scalar=self.nelem == 1 and len(self._shape) == 0)

    def asnumpy(self, flush=True):
        if self.nelem == 0:
            raise RuntimeError("The size of the zero!")
        if flush:
            _bh_api.flush()
        data = _bh_api.data_get(self.base._bh_dtype_enum, self._bhc_handle, True, True, False, self.base.nbytes)
        ret = np.frombuffer(data, dtype=self.dtype, offset=self.offset * self.base.itemsize)
        return np.lib.stride_tricks.as_strided(ret, self._shape, [s * self.base.itemsize for s in self._strides])

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
        ret = BhArray(self._shape, dtype)
        assign(self, ret)
        return ret

    def isscalar(self):
        return len(self._shape) == 0 and self.nelem == 1

    def isbehaving(self):
        return self.offset == 0 and self.iscontiguous()

    def empty(self):
        return self.nelem == 0 and not self.isscalar()

    def iscontiguous(self):
        acc = 1
        for shape, stride in zip(reversed(self._shape), reversed(self._strides)):
            if shape > 1 and stride != acc:
                return False
            else:
                acc *= shape
        return True

    def copy(self):
        """Return a copy of the array.

        Returns
        -------
        out : BhArray
            Copy of `self`
        """
        return self.astype(self.dtype, always_copy=True)

    def transpose(self, axes=None):
        """Permute the dimensions of an array.

        Parameters
        ----------
        axes : list of ints, optional
            By default, reverse the dimensions, otherwise permute the axes
            according to the values given.
        """
        if axes is None:
            axes = list(reversed(range(len(self._shape))))

        ret = self.view()
        ret._shape = tuple([self._shape[i] for i in axes])
        ret._strides = tuple([self._strides[i] for i in axes])
        return ret

    def flatten(self, always_copy=True):
        """ Return a copy of the array collapsed into one dimension.

        Parameters
        ----------
        always_copy : boolean
            When False, a copy is only made when necessary

        Returns
        -------
        y : ndarray
            A copy of the array, flattened to one dimension.

        Notes
        -----
        The order of the data in memory is always row-major (C-style).

        Examples
        --------
        >>> a = np.array([[1,2], [3,4]])
        >>> a.flatten()
        array([1, 2, 3, 4])
        """
        shape = (self.nelem,)
        if not self.iscontiguous():
            assert (self.copy().iscontiguous())
            ret = self.copy().flatten(always_copy=False)  # copy() makes the array contiguous
            assert (ret.iscontiguous())
            return ret
        else:
            ret = BhArray(shape, self.dtype, offset=self.offset, base=self.base)
            if always_copy:
                return ret.copy()
            else:
                return ret

    def ravel(self):
        """ Return a contiguous flattened array.

        A 1-D array, containing the elements of the input, is returned. A copy is made only if needed.

        Returns
        -------
        y : ndarray
            A copy or view of the array, flattened to one dimension.
        """
        return self.flatten(always_copy=False)

    def reshape(self, shape):
        from .ufuncs import assign
        length = util.total_size(shape)
        if length != self.nelem:
            raise RuntimeError("Total size cannot change when reshaping")

        flat = self.flatten()
        return BhArray(shape, flat.dtype, base=flat.base)

    def __getitem_at_dim(self, dim, key):
        if np.isscalar(key):
            if not isinstance(key, _dtype_util.integers):
                raise IndexError("Only integers, slices (`:`), ellipsis (`...`), np.newaxis (`None`) and "
                                 "integer or boolean arrays are valid indices")
            if len(self._shape) <= dim or key >= self._shape[dim]:
                raise IndexError("Index out of bound")
            shape = list(self._shape)
            shape.pop(dim)
            strides = list(self._strides)
            strides.pop(dim)
            offset = self.offset + key * self._strides[dim]
            return BhArray(shape, self.dtype, offset=offset, strides=strides, base=self.base, is_scalar=len(shape) == 0)
        elif isinstance(key, slice):
            if len(self._shape) <= dim:
                raise IndexError("IndexError: too many indices for array")
            length = self._shape[dim]
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
            # noinspection PyTypeChecker
            shape = list(self._shape[:dim]) + [new_length] + list(self._shape[dim + 1:])
            strides = list(self._strides[:dim]) + [step * self._strides[dim]] + list(self._strides[dim + 1:])
            offset = self.offset + start * self._strides[dim]
            return BhArray(shape, self.dtype, offset=offset, strides=strides, base=self.base)
        elif key is None:
            shape = list(self._shape)
            shape.insert(dim, 1)
            strides = list(self._strides)
            strides.insert(dim, 0)
            return BhArray(shape, self.dtype, offset=self.offset, strides=strides, base=self.base)
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
                while len(key) < len(self._shape) + 1:
                    key.insert(idx + 1, slice(None, None, None))
                key.pop(idx)
                assert (len(key) == len(self._shape))

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

    def __setitem__(self, key, value):
        from .ufuncs import assign
        assign(value, self.__getitem__(key))

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
