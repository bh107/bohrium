# -*- coding: utf-8 -*-
import os
import math
import numpy as np
# noinspection PyProtectedMember,PyUnresolvedReferences
from bohrium_api import _bh_api
from . import _dtype_util, util


def implements(numpy_function):
    """Register an __array_function__ implementation for BhArray objects."""
    def decorator(func):
        BhArray._NP_FUNCTIONS[numpy_function] = func
        return func
    return decorator


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
    _NP_FUNCTIONS = {}

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

    # creation methods
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

    # destructor
    def __del__(self):
        if hasattr(self, '_bhc_handle') and self._bhc_handle is not None:
            _bh_api.destroy(self.base._bh_dtype_enum, self._bhc_handle)

    # properties
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

    # NumPy interfaces
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        from .ufuncs import ufunc_dict
        cls = self.__class__
        ufunc_name = ufunc.__name__
        if ufunc_name in ufunc_dict:
            inputs = (cls.from_object(i) if not isinstance(i, cls) else i for i in inputs)
            return getattr(ufunc_dict[ufunc_name], method)(*inputs, **kwargs)
        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        cls = self.__class__
        if func not in cls._NP_FUNCTIONS:
            args = (arg.asnumpy() if isinstance(arg, cls) else arg for arg in args)
            return cls.from_numpy(func(*args, **kwargs))
        args = (cls.from_object(arg) if not isinstance(arg, cls) else arg for arg in args)
        return cls._NP_FUNCTIONS[func](*args, **kwargs)

    @property
    def __array_interface__(self):
        """Exposing the The Array Interface <https://docs.scipy.org/doc/numpy/reference/arrays.interface.html>"""
        if self.nelem == 0:
            raise RuntimeError("The size of the zero!")
        _bh_api.flush()
        typestr = np.dtype(self.base.dtype).str
        shape = self.shape
        strides = tuple(s * self.base.itemsize for s in self.strides)
        data_ptr = _bh_api.data_get(self.base._bh_dtype_enum, self._bhc_handle, True, True, False, self.base.nbytes)
        data = (data_ptr+self.offset * self.base.itemsize, False)  # read-only is false
        return dict(typestr=typestr, shape=shape, strides=strides, data=data, version=0)

    def asnumpy(self):
        """Returns a NumPy array that points to the same memory as this BhArray"""
        return np.array(self, copy=False)

    def copy2numpy(self):
        """Returns a NumPy array that is a copy of this BhArray"""
        if self.nelem == 0:
            return np.array([], self.dtype)
        else:
            return self.asnumpy().copy()

    def view(self):
        """Returns a new view that points to the same base as this BhArray"""
        return BhArray(self._shape, self.dtype, self._strides, self.offset, self.base,
                       is_scalar=self.nelem == 1 and len(self._shape) == 0)

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
            ret = self.copy().flatten(always_copy=False)  # copy() makes the array contiguous
            assert (ret.iscontiguous())
            return ret
        else:
            ret = BhArray(shape, self.dtype, offset=self.offset, base=self.base)
            if always_copy:
                return ret.copy()
            else:
                return ret

    def reshape(self, shape):
        length = util.total_size(shape)
        if length != self.nelem:
            raise RuntimeError("Total size cannot change when reshaping")

        flat = self.flatten()
        return BhArray(shape, flat.dtype, base=flat.base)

    def copy(self):
        return self.astype(self.dtype, always_copy=True)

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

    # maths
    def __add__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['add'](self, other)

    def __radd__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['add'](other, self)

    def __iadd__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['add'](self, other, self)

    def __sub__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['subtract'](self, other)

    def __rsub__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['subtract'](other, self)

    def __isub__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['subtract'](self, other, self)

    def __mul__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['multiply'](self, other)

    def __rmul__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['multiply'](other, self)

    def __imul__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['multiply'](self, other, self)

    def __floordiv__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['floor_divide'](self, other)

    def __rfloordiv__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['floor_divide'](other, self)

    def __ifloordiv__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['floor_divide'](self, other, self)

    def __truediv__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['true_divide'](self, other)

    def __rtruediv__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['true_divide'](other, self)

    def __itruediv__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['true_divide'](self, other, self)

    def __div__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['divide'](self, other)

    def __rdiv__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['divide'](other, self)

    def __idiv__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['divide'](self, other, self)

    def __mod__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['mod'](self, other)

    def __rmod__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['mod'](other, self)

    def __imod__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['mod'](self, other, self)

    def __pow__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['power'](self, other)

    def __rpow__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['power'](other, self)

    def __ipow__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['power'](self, other, self)

    # logic
    def __and__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['bitwise_and'](self, other)

    def __rand__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['bitwise_and'](other, self)

    def __iand__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['bitwise_and'](self, other, self)

    def __xor__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['bitwise_xor'](self, other)

    def __rxor__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['bitwise_xor'](other, self)

    def __ixor__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['bitwise_xor'](self, other, self)

    def __or__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['bitwise_or'](self, other)

    def __ror__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['bitwise_or'](other, self)

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

    # Comparison
    def __lt__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['less'](self, other)

    def __le__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['less_equal'](self, other)

    def __eq__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['equal'](self, other)

    def __ne__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['not_equal'](self, other)

    def __gt__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['greater'](self, other)

    def __ge__(self, other):
        from .ufuncs import ufunc_dict
        return ufunc_dict['greater_equal'](self, other)

    # string representations
    def __str__(self):
        if self.nelem == 0:
            return "[]"
        else:
            return str(self.asnumpy())

    def __repr__(self):
        np_repr = self.asnumpy().__repr__()
        return self.__class__.__name__ + np_repr[5:].replace('\n', '\n  ')


# NumPy functions
@implements(np.mean)
def mean(a, axis=None, dtype=None, out=None):
    import warnings
    from .ufuncs import ufunc_dict

    add = ufunc_dict['add']

    def _count_reduce_items(arr, axis):
        if axis is None:
            axis = tuple(range(arr.ndim))
        if not isinstance(axis, tuple):
            axis = (axis,)
        items = 1
        for ax in axis:
            items *= arr.shape[ax]
        return items

    def _mean(arr, axis=None, dtype=None, out=None):
        is_float16_result = False
        rcount = _count_reduce_items(arr, axis)
        # Make this warning show up first
        if rcount == 0:
            warnings.warn("Mean of empty slice.", RuntimeWarning, stacklevel=2)

        # Cast bool, unsigned int, and int to float64 by default
        if dtype is None:
            if issubclass(arr.dtype, (np.integer, np.bool_)):
                dtype = np.dtype('f8')
            elif issubclass(arr.dtype, np.float16):
                dtype = np.dtype('f4')
                is_float16_result = True

        ret = add.reduce(arr, axis=axis, dtype=dtype, out=out)
        if ret.isscalar():
            ret = ret.dtype(ret)
        ret /= rcount
        if is_float16_result and out is None:
            ret = a.dtype(ret)
        return ret

    return _mean(a, axis=axis, dtype=dtype, out=out)


@implements(np.copy)
def copy(arr):
    """Return a copy of the array.

    Returns
    -------
    out : BhArray
        Copy of `arr`
    """
    return arr.copy()


@implements(np.transpose)
def transpose(arr, axes=None):
    """Permute the dimensions of an array.

    Parameters
    ----------
    axes : list of ints, optional
        By default, reverse the dimensions, otherwise permute the axes
        according to the values given.
    """
    if axes is None:
        axes = list(reversed(range(len(arr._shape))))

    ret = arr.view()
    ret._shape = tuple([arr._shape[i] for i in axes])
    ret._strides = tuple([arr._strides[i] for i in axes])
    return ret


@implements(np.ravel)
def ravel(arr):
    """ Return a contiguous flattened array.

    A 1-D array, containing the elements of the input, is returned. A copy is made only if needed.

    Returns
    -------
    y : ndarray
        A copy or view of the array, flattened to one dimension.
    """
    return arr.flatten(always_copy=False)


@implements(np.reshape)
def reshape(arr, shape):
    return arr.reshape(shape)
