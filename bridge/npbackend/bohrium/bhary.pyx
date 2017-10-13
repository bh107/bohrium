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
import sys
from ._util import dtype_equal, dtype_support, dtype_in
from . import target
from . import numpy_backport
import operator
import functools
import numpy_force as numpy
cimport numpy as numpy

cpdef get_cdata(numpy.ndarray ary):
    """Returns the array data pointer as an integer. 
    This function is MUCH faster than the ndarray.ctypes attribute"""
    return <unsigned long long> ary.data

def check(ary):
    """Returns True if 'ary' is a Bohrium array"""

    try:
        #This will fail if the base is a NumPy array
        base = get_base(ary)
    except AttributeError:
        base = ary
    return hasattr(base, "bhc_ary")

def check_biclass_np_over_bh(ary):
    """Returns True if 'ary' is a NumPy view with a Bohrium base array"""

    if not isinstance(ary, numpy.ndarray):
        return False
    try:
        if not check(get_base(ary)):
            return False
    except AttributeError:
        return False

    from . import _bh  #We import locally in order to avoid import cycle
    return not isinstance(ary, _bh.ndarray)

def check_biclass_bh_over_np(ary):
    """Returns True if 'ary' is a Bohrium view with a NumPy base array"""

    return hasattr(ary, "bhc_ary") and not check(get_base(ary))

def fix_biclass(ary):
    """
    Makes sure that when 'ary' or its base is a Bohrium array, both of them are.
    """

    if not isinstance(ary, numpy.ndarray):
        return ary

    if check_biclass_np_over_bh(ary):
        return ary.view(type(get_base(ary)))
    elif check_biclass_bh_over_np(ary):
        from . import array_create
        ary = array_create.array(ary, bohrium=False)
        return array_create.array(ary, bohrium=True)
    else:
        return ary

def fix_biclass_wrapper(func):
    """
    Function decorator that makes sure that the function doesn't reads or writes biclass arrays
    """

    if hasattr(func, "_wrapped_fix_biclass"):
        return func

    def inner(*args, **kwargs):
        """Invokes 'func' and strips "biclass" from the result."""

        # Normally, we checks all arguments for biclass arrays, but this can be disabled individually
        # by setting the function keyword argument 'fix_biclass' to False
        if kwargs.get("fix_biclass", True) and sys.version_info[0] < 3:
            args = [fix_biclass(a) for a in args]
        if "fix_biclass" in kwargs:
            del kwargs["fix_biclass"]
        ret = func(*args, **kwargs)
        return fix_biclass(ret)

    try:
        #Flag that this function has been handled
        setattr(inner, "_wrapped_fix_biclass", True)
    except:  #In older versions of cython, this is not possible
        pass
    return inner

def new(shape, dtype, bhc_ary=None):
    """
    Creates a new bohrium.ndarray with 'bhc_ary' as the Bohrium-C part.
    Use a new Bohrium-C array when 'bhc_ary' is None.
    """

    from . import _bh  #We import locally in order to avoid import cycle
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
     * offset
     * shape
     * strides

    NOTE: The 'base' of the view is allowed to be different.
    """

    if view1.dtype != view2.dtype:
        return False

    view1_ndim = view1.ndim if view1.ndim > 0 else 1
    view2_ndim = view2.ndim if view2.ndim > 0 else 1

    if view1_ndim != view2_ndim:
        return False
    if list(view1.shape) != list(view2.shape):
        return False
    if list(view1.strides) != list(view2.strides):
        return False

    b1 = get_base(view1)
    b2 = get_base(view1)
    v1_offset = view1.start if hasattr(view1, 'start') else get_cdata(view1) - get_cdata(b1)
    v2_offset = view2.start if hasattr(view2, 'start') else get_cdata(view2) - get_cdata(b2)
    if v1_offset != v2_offset:
        return False
    return True

def in_bhmem(ary):
    """Returns True when 'ary' is in the memory address space of Bohrium"""

    return get_base(ary).bhc_ary is not None

def get_bhc(ary):
    """
    Returns the Bohrium-C part of the array (supports both Bohrium or NumPy arrays)
    NB: the returned object is always a view
    """

    base = get_base(ary)
    # Lets see if we can use an already existing array-view
    if hasattr(ary, 'bhc_view') and ary.bhc_view is not None:
        if base.bhc_ary_version == ary.bhc_view_version:
            if not identical_views(ary, ary.bhc_view):
                ary.bhc_view = None
            else:
                return ary.bhc_view

    if not check(base):
        raise TypeError("the base must be a Bohrium array")
    if not ary.flags['BEHAVED']:
        raise ValueError("Bohrium arrays must be aligned, writeable, and in machine byte-order")
    if not dtype_equal(ary, base):
        # If 'ary' is real or imag view of 'base', we will convert the view into a real base array
        if dtype_in(ary.dtype, [numpy.float32, numpy.float64]) and \
                dtype_in(base.dtype, [numpy.complex64, numpy.complex128]):

            # All this is simply a hack to reinterpret 'ary' as a complex view of the 'base'
            offset = (get_cdata(ary) - get_cdata(base)) // base.itemsize
            cary = numpy.frombuffer(base, dtype=base.dtype, offset=offset * base.itemsize)
            cary = numpy_backport.as_strided(cary, ary.shape, ary.strides)

            # if the view/base offset is aligned with the complex dtype, we know that the
            # 'ary' is a view of the real part of 'base'
            from . import ufuncs
            if (get_cdata(ary) - get_cdata(base)) % base.itemsize == 0:
                ary = ufuncs.real(cary)
            else:
                ary = ufuncs.imag(cary)
            base = ary  # 'ary' is now itself a base array
        else:
            raise ValueError("Bohrium base and view must have same data type")
    if not dtype_support(ary.dtype):
        raise ValueError("Bohrium does not support the data type: %s" % ary.dtype)

    if 0 in ary.shape:  #Lets use a dummy strides and offset for zero-sized views
        strides = [0] * ary.ndim
        offset = 0
    else:
        if not get_cdata(base) <= get_cdata(ary) < get_cdata(base) + base.nbytes:
            raise ValueError("The view must point to data within the base array")

        # Lets make sure that 'ary' has a Bohrium-C base array
        if base.bhc_ary is None:
            base._data_np2bhc()

        offset = (get_cdata(ary) - get_cdata(base)) // base.itemsize
        if (get_cdata(ary) - get_cdata(base)) % base.itemsize != 0:
            raise TypeError("The view offset must be element aligned")
        if not 0 <= offset < base.size:
            raise TypeError("The view offset is greater than the total number of elements in the base!")
        strides = []
        for stride in ary.strides:
            strides.append(stride // base.itemsize)
            if stride % base.itemsize != 0:
                raise TypeError("The strides must be element aligned")

    ndim = ary.ndim if ary.ndim > 0 else 1
    shape = ary.shape if len(ary.shape) > 0 else (1,)
    strides = strides if len(strides) > 0 else (1,)

    ret = target.View(ndim, offset, shape, strides, base.bhc_ary)
    if hasattr(ary, 'bhc_view'):
        ary.bhc_view = ret
        ary.bhc_view_version = base.bhc_ary_version
    return ret

def del_bhc(ary):
    """Delete the Bohrium-C part of the bohrium.ndarray and its base."""

    if ary.bhc_ary is not None:
        ary.bhc_ary = None
    if ary.bhc_view is not None:
        ary.bhc_view = None
    base = get_base(ary)
    if base is not ary:
        del_bhc(base)
    else:
        base.bhc_ary_version += 1

def get_bhc_data_pointer(ary, copy2host=True, allocate=False, nullify=False):
    """
    Return the Bohrium-C data pointer (represented by a Python integer)
    When allocate is True, it allocates memory instead of returning None
    When nullify is True, the Bohrium-C data pointer is set to NULL
    """

    if not check(ary):
        raise TypeError("must be a Bohrium array")
    ary = get_base(ary)
    return target.get_data_pointer(get_bhc(ary), copy2host, allocate, nullify)

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
