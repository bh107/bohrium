#!/usr/bin/env python
"""
This module consist of bohrium.ndarray methods

The functions here serve as a means to determine whether a given
array is a numpy.ndarray or a bohrium.ndarray as well as moving
between the two "spaces".

--- License ---
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
from ._util import dtype_equal
from . import target
import operator
import functools

def check(ary):
    """Returns True if 'ary' is a Bohrium array"""

    try:
        #This will fail if the base is a NumPy array
        base = get_base(ary)
    except AttributeError:
        base = ary
    return hasattr(base, "bhc_ary")

def check_biclass(ary):
    """Returns True if 'ary' is a NumPy view with a Bohrium base array"""

    try:
        if not check(get_base(ary)):
            return False
    except AttributeError:
        return False

    import _bh
    return not isinstance(ary, _bh.ndarray)

def fix_biclass(ary):
    """
    Returns a Bohrium version of 'ary' if 'ary' is a NumPy view with a
    Bohrium base array else 'ary' is returned unmodified
    """

    if check_biclass(ary):
        return ary.view(type(get_base(ary)))
    else:
        return ary

def fix_returned_biclass(func):
    """
    Function decorator that makes sure that the function doesn't return a
    biclass.
    """

    def inner(*args, **kwargs):
        """Invokes 'func' and strips "biclass" from the result."""
        ret = func(*args, **kwargs)
        return fix_biclass(ret)

    return inner

def new(shape, dtype, bhc_ary=None):
    """
    Creates a new bohrium.ndarray with 'bhc_ary' as the Bohrium-C part.
    Use a new Bohrium-C array when 'bhc_ary' is None.
    """

    import _bh #We import locally in order to avoid cycles
    ret = _bh.ndarray(shape, dtype=dtype)
    if bhc_ary is None:
        new_bhc_base(ret)
    else:
        ret.bhc_ary = bhc_ary
    return ret

def new_bhc_base(ary):
    """Creates a new Bohrium-C base array."""

    if not check(ary):
        raise TypeError("must be a Bohrium array")

    if ary.base is not None:
        raise ValueError("Views do not have a Bohrium-C base array")

    if ary.bhc_ary is not None:
        raise ValueError("The array has a Bohrium-C base array already")

    if not ary.flags['BEHAVED']:
        raise ValueError("Bohrium arrays must be aligned, writeable, and in machine byte-order")

    if not ary.flags['OWNDATA']:
        raise ValueError("Bohrium base arrays must own its data")

    if not ary.flags['C_CONTIGUOUS']:
        raise ValueError("For now Bohrium only supports C-style arrays")
    shape = ary.shape if len(ary.shape) > 0 else (1,)
    totalsize = functools.reduce(operator.mul, shape, 1)
    ary.bhc_ary = target.Base(totalsize, ary.dtype)

def get_base(ary):
    """Get the final base array of 'ary'."""

    if ary.base is None:
        return ary
    else:
        base = ary.base
        while base.base is not None:
            base = base.base
        return base

def is_base(ary):
    """Return True when 'ary' is a base array."""

    base = get_base(ary)
    return base is ary

def identical_views(view1, view2):
    """
    Returns True when 'view1' equals 'view2'.

    Equality is defined as having identical:

     * dtype
     * ndim
     * shape
     * strides

    NOTE: The 'base' of the view is allowed to be different.
    """

    if view1.dtype != view2.dtype:
        return False
    if view1.ndim != view2.ndim:
        return False
    if list(view1.shape) != list(view2.shape):
        return False
    if list(view1.strides) != list(view2.strides):
        return False
    return True

def get_bhc(ary):
    """
    Returns the Bohrium-C part of the array (supports both Bohrium or NumPy arrays)
    NB: the returned object is always a view
    """

    # Lets see if we can use an already existing array-view
    if hasattr(ary, 'bhc_view') and ary.bhc_view is not None:
        if not identical_views(ary, ary.bhc_view):
            ary.bhc_view = None
        else:
            return ary.bhc_view

    base = get_base(ary)
    if not check(base):
        raise TypeError("the base must be a Bohrium array")
    if not ary.flags['BEHAVED']:
        raise ValueError("Bohrium arrays must be aligned, writeable, and in machine byte-order")
    if not dtype_equal(ary, base):
        raise ValueError("Bohrium base and view must have same data type")
    if not base.ctypes.data <= ary.ctypes.data < base.ctypes.data + base.nbytes:
        raise ValueError("The view must point to data within the base array")

    # Lets make sure that 'ary' has a Bohrium-C base array
    if base.bhc_ary is None:
        base._data_np2bhc()

    offset = (ary.ctypes.data - base.ctypes.data) / base.itemsize
    if (ary.ctypes.data - base.ctypes.data) % base.itemsize != 0:
        raise TypeError("The view offset must be element aligned")
    if not 0 <= offset < base.size:
        raise TypeError("The view offset is greater than the total number of elements in the base!")
    strides = []
    for stride in ary.strides:
        strides.append(stride / base.itemsize)
        if stride % base.itemsize != 0:
            raise TypeError("The strides must be element aligned")

    ndim = ary.ndim if ary.ndim > 0 else 1
    shape = ary.shape if len(ary.shape) > 0 else (1,)
    strides = strides if len(strides) > 0 else (1,)

    ret = target.View(ndim, offset, shape, strides, base.bhc_ary)
    if hasattr(ary, 'bhc_view'):
        ary.bhc_view = ret
    return ret

def del_bhc(ary):
    """Delete the Bohrium-C part of the bohrium.ndarray and its base."""

    if ary.bhc_ary is not None:
        ary.bhc_ary = None
    base = get_base(ary)
    if base is not ary:
        del_bhc(base)

def get_bhc_data_pointer(ary, allocate=False, nullify=False):
    """
    Return the Bohrium-C data pointer (represented by a Python integer)
    When allocate is True, it allocates memory instead of returning None
    When nullify is True, the Bohrium-C data pointer is set to NULL
    """

    if not check(ary):
        raise TypeError("must be a Bohrium array")
    ary = get_base(ary)
    return target.get_data_pointer(get_bhc(ary), allocate, nullify)

def set_bhc_data_from_ary(self, ary):
    """
    Copies the NumPy part of 'ary' to the Bohrium-C part of 'self'
    NB: the NumPy part of 'ary' must not be mprotect'ed and can be a regular numpy.ndarray array
    """

    if not check(self):
        raise TypeError("must be a Bohrium array")
    ary = get_base(ary)
    if not ary.flags['BEHAVED']:
        raise ValueError("Input array must be aligned, writeable, and in machine byte-order")
    if not ary.flags['C_CONTIGUOUS']:
        raise ValueError("Input array must be C-style contiguous")

    target.set_bhc_data_from_ary(get_bhc(self), ary)

