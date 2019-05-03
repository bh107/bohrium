# -*- coding: utf-8 -*-
"""
==========================
Useful Data Type Functions
==========================
"""

import numpy as np
from bohrium_api import _bh_api

_size_of_dtype_in_bytes = {
    np.bool: 1,
    np.int8: 1,
    np.int16: 2,
    np.int32: 4,
    np.int64: 8,
    np.uint8: 1,
    np.uint16: 2,
    np.uint32: 4,
    np.uint64: 8,
    np.float32: 4,
    np.float64: 8,
    np.complex64: 8,
    np.complex128: 16,
}

_dtype_np2bh_enum = {
    np.bool: _bh_api.bool,
    np.int8: _bh_api.int8,
    np.int16: _bh_api.int16,
    np.int32: _bh_api.int32,
    np.int64: _bh_api.int64,
    np.uint8: _bh_api.uint8,
    np.uint16: _bh_api.uint16,
    np.uint32: _bh_api.uint32,
    np.uint64: _bh_api.uint64,
    np.float32: _bh_api.float32,
    np.float64: _bh_api.float64,
    np.complex64: _bh_api.complex64,
    np.complex128: _bh_api.complex128,
}

_dtype_type_to_dtype = {
    np.bool: np.bool,
    np.int8: np.int8,
    np.int16: np.int16,
    np.int32: np.int32,
    np.int64: np.int64,
    np.uint8: np.uint8,
    np.uint16: np.uint16,
    np.uint32: np.uint32,
    np.uint64: np.uint64,
    np.float32: np.float32,
    np.float64: np.float64,
    np.complex64: np.complex64,
    np.complex128: np.complex128,
    "bool": np.bool,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "float32": np.float32,
    "float64": np.float64,
    "complex64": np.complex64,
    "complex128": np.complex128,
    np.dtype("bool"): np.bool,
    np.dtype("int8"): np.int8,
    np.dtype("int16"): np.int16,
    np.dtype("int32"): np.int32,
    np.dtype("int64"): np.int64,
    np.dtype("uint8"): np.uint8,
    np.dtype("uint16"): np.uint16,
    np.dtype("uint32"): np.uint32,
    np.dtype("uint64"): np.uint64,
    np.dtype("float32"): np.float32,
    np.dtype("float64"): np.float64,
    np.dtype("complex64"): np.complex64,
    np.dtype("complex128"): np.complex128,
    _bh_api.bool: np.bool,
    _bh_api.int8: np.int8,
    _bh_api.int16: np.int16,
    _bh_api.int32: np.int32,
    _bh_api.int64: np.int64,
    _bh_api.uint8: np.uint8,
    _bh_api.uint16: np.uint16,
    _bh_api.uint32: np.uint32,
    _bh_api.uint64: np.uint64,
    _bh_api.float32: np.float32,
    _bh_api.float64: np.float64,
    _bh_api.complex64: np.complex64,
    _bh_api.complex128: np.complex128,
    bool: np.bool,
    int: np.int64,
    float: np.float64,
    complex: np.complex128,
}

# In Python 2.7 we should also map the `long` type
try:
    _dtype_type_to_dtype[long] = np.int64
except NameError:
    pass

# In Python 3 `str` is the basestring
try:
    # noinspection PyCompatibility
    basestring
except NameError:
    # noinspection PyShadowingBuiltins
    basestring = str

try:
    integers = (int, long)
except NameError:
    integers = (int,)  # `long` is not int Python3


def type_to_dtype(any_type):
    return _dtype_type_to_dtype[any_type]


def obj_to_dtype(obj):
    if isinstance(obj, (np.dtype, basestring, type)):
        return type_to_dtype(obj)
    elif hasattr(obj, "dtype"):
        return type_to_dtype(obj.dtype)
    else:
        return type_to_dtype(type(obj))


def size_of(dtype):
    return _size_of_dtype_in_bytes[dtype]


def np2bh_enum(dtype):
    """Convert data type from Bohrium to NumPy"""
    return _dtype_np2bh_enum[dtype]
