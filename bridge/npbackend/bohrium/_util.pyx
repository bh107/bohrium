#!/usr/bin/env python
"""
This file is part of Bohrium and copyright (c) 2012 the Bohrium
http://bohrium.bitbucket.org

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
"""
import collections
import numpy_force as np
from bohrium_api import _info

def dtype_of(obj):
    """Returns the dtype of 'obj'."""

    if isinstance(obj, np.dtype):
        return obj
    elif isinstance(obj, basestring):
        tmp = obj
    elif isinstance(obj, type):
        tmp = obj
    elif hasattr(obj, "dtype"):
        tmp = obj.dtype
    else:
        tmp = type(obj)
    return np.dtype(tmp)

def dtype_equal(*args):
    """Returns True when all 'args' is the same dtype."""

    if len(args) > 1:
        first_tmp = dtype_of(args[0])
        for tmp in args[1:]:
            if first_tmp is not dtype_of(tmp):
                return False
    return True

def dtype_in(dtype, dtypes):
    """Returns True when 'dtype' is in the list of 'dtypes'."""

    for datatype in dtypes:
        if dtype_equal(dtype, datatype):
            return True
    return False

def dtype_is_float(dtype):
    """Returns True when 'dtype' is a float or complex."""

    return dtype_in(dtype, [np.float32, np.float64, np.complex64, np.complex128])

def dtype_name(obj):
    """Returns the Bohrium name of the data type of the object 'obj'."""

    dtype = dtype_of(obj)
    if dtype_in(dtype, [np.bool_, np.bool, bool]):
        return 'bool8'
    else:
        return dtype.name

def type_sig(op_name, inputs):
    """
    Returns the type signature (output, input) to use with the given operation.
    NB: we only returns the type of the first input thus all input types must
        be identical
    """

    func = _info.op[op_name]
    #Note that we first use the dtype before the array as inputs to result_type()
    inputs = [getattr(t, 'dtype', t) for t in inputs]
    dtype = np.result_type(*inputs)
    for sig in func['type_sig']:
        if dtype.name == sig[1]:
            return (np.dtype(sig[0]), np.dtype(sig[1]))

    # Let's try use a float signature for the integer input
    if np.issubdtype(dtype, np.integer):
        for sig in func['type_sig']:
            if 'float' in sig[1]:
                return (np.dtype(sig[0]), np.dtype(sig[1]))

    raise TypeError("The ufunc bohrium.%s() does not support input data type: %s." % (op_name, dtype.name))

def dtype_support(dtype):
    """Returns True when Bohrium supports 'dtype' """

    if dtype_in(dtype, _info.numpy_types()):
        return True
    else:
        return False

def totalsize(array_like):
    """Return the total size of an list like object such as a bharray, ndarray, list, etc."""
    if np.isscalar(array_like):
        return 1
    elif hasattr(array_like, "size"):
        return array_like.size
    elif isinstance(array_like, collections.Iterable) and not isinstance(array_like, basestring):
        return sum(totalsize(item) for item in array_like)
    else:
        return 1

def is_scalar(a):
    """Is `a` a scalar type or 0-dim array?"""
    return np.isscalar(a) or a.ndim == 0