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
    from . import _bh  #We import locally in order to avoid import cycle
    return isinstance(base, _bh.ndarray)

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
