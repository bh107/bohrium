#!/usr/bin/env python
"""
/*
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
*/
"""
from _util import dtype_from_bhc, dtype_name
import bhc

# This module consist of bohrium.ndarray methods

#Returns True if 'ary' is a Bohrium array
def check(ary):
    return hasattr(ary, "bhc_ary")

#Creates a new bohrium.ndarray with 'bhc_ary' as the Bohrium-C part.
#Use a new Bohrium-C array when 'bhc_ary' is None.
def new(shape, dtype, bhc_ary=None):
    import _bh #We import locally in order to avoid cycles
    ret = _bh.ndarray(shape, dtype=dtype)
    if bhc_ary is None:
        new_bhc_base(ret)
    else:
        ret.bhc_ary = bhc_ary
    exec "bhc.bh_multi_array_%s_set_temp(ret.bhc_ary, 0)"%dtype_name(dtype)
    return ret

#Creates a new Bohrium-C base array.
def new_bhc_base(ary):
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

    dtype = dtype_name(ary)
    exec "ary.bhc_ary = bhc.bh_multi_array_%s_new_empty(ary.ndim, ary.shape)"%dtype

#Get the final base array of 'ary'
def get_base(ary):
    if ary.base is None:
        return ary
    else:
        base = ary.base
        while base.base is not None:
            base = base.base
        return base

#Return True when 'ary' is a base array
def is_base(ary):
    b = get_base(ary)
    return b is ary

#Returns the Bohrium-C part of the bohrium.ndarray
def get_bhc(ary):
    if not check(ary):
        raise TypeError("must be a Bohrium array")

    #Find the base array and make sure it has a Bohrium-C base array
    base = get_base(ary)
    if base.bhc_ary is None:
        base._data_np2bhc()

    if ary is base:#We a returning a base array
        return base.bhc_ary
    elif ary.bhc_ary is None:
        if not ary.flags['BEHAVED']:
            raise ValueError("Bohrium arrays must be aligned, writeable, and in machine byte-order")
        dtype = dtype_name(ary)
        exec "bh_base = bhc.bh_multi_array_%s_get_base(base.bhc_ary)"%dtype
        offset = (ary.ctypes.data - base.ctypes.data) / base.itemsize
        if (ary.ctypes.data - base.ctypes.data) % base.itemsize != 0:
            raise TypeError("The view offset must be element aligned")
        if not (0 <= offset < base.size):
            raise TypeError("The view offset is greater than the total number of elements in the base!")
        strides = []
        for s in ary.strides:
            strides.append(s / base.itemsize)
            if s % base.itemsize != 0:
                raise TypeError("The strides must be element aligned")

        exec "ary.bhc_ary = bhc.bh_multi_array_%s_new_from_view(bh_base, ary.ndim, "\
                            "offset, ary.shape, strides)"%dtype
    return ary.bhc_ary

#Delete the Bohrium-C object
def del_bhc_obj(bhc_obj):
    exec "bhc.bh_multi_array_%s_destroy(bhc_obj)"%dtype_from_bhc(bhc_obj)

#Delete the Bohrium-C part of the bohrium.ndarray and its base
def del_bhc(ary):
    if ary.bhc_ary is not None:
        del_bhc_obj(ary.bhc_ary)
        ary.bhc_ary = None
    base = get_base(ary)
    if base is not ary:
        del_bhc(base)

#Return the Bohrium-C data pointer (represented by a Python integer)
#When allocate is True, it allocates memory instead of returning None
def get_bhc_data_pointer(ary, allocate=False, nullify=False):
    if not check(ary):
        raise TypeError("must be a Bohrium array")
    ary = get_base(ary)
    dtype = dtype_name(ary)
    bhc_ary = get_bhc(ary)
    exec "bhc.bh_multi_array_%s_sync(bhc_ary)"%dtype
    exec "base = bhc.bh_multi_array_%s_get_base(bhc_ary)"%dtype
    exec "data = bhc.bh_multi_array_%s_get_base_data(base)"%dtype
    if data is None:
        if not allocate:
            return 0
        exec "data = bhc.bh_multi_array_%s_get_base_data_and_force_alloc(base)"%dtype
        if data is None:
            raise MemoryError()
    if nullify:
        exec "bhc.bh_multi_array_%s_nullify_base_data(base)"%dtype
    return int(data)

