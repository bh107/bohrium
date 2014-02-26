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
import _util
import bhc

# This module consist of bohrium.ndarray methods

#Creates a new Bohrium-C base array.
def new_bhc_base(ary):
    if not hasattr(ary, "bhc_ary"):
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

    dtype = _util.dtype_name(ary)
    size = ary.size
    exec "ary.bhc_ary = bhc.bh_multi_array_%s_new_empty(1, (%d,))"%(dtype,size)

#Get the final base array of 'ary'
def get_base(ary):
    if ary.base is None:
        return ary
    else:
        base = ary.base
        while base.base is not None:
            base = base.base
        return base

#Returns the Bohrium-C array
def get_bhc(ary):
    if not hasattr(ary, "bhc_ary"):
        raise TypeError("must be a Bohrium array")

    #Find the base array and make sure it has a Bohrium-C base array
    base = get_base(ary)
    if base.bhc_ary is None:
        new_bhc_base(base)
        data_np2bhc(base)

    if ary is base:#We a returning a base array
        return base.bhc_ary
    else:
        raise NotImplementedError("TODO: handle views")

#Return the Bohrium-C data pointer (represented by a Python integer)
def get_bhc_data_pointer(ary):
    if not hasattr(ary, "bhc_ary"):
        raise TypeError("must be a Bohrium array")
    ary = get_base(ary)
    dtype = _util.dtype_name(ary)
    bhc_ary = get_bhc(ary)
    exec "bhc.bh_multi_array_%s_sync(bhc_ary)"%dtype
    exec "base = bhc.bh_multi_array_%s_get_base(bhc_ary)"%dtype
    exec "data = bhc.bh_multi_array_%s_get_base_data(base)"%dtype
    if data is None:
        return 0
    else:
        return int(data)

def data_np2bhc(ary):
    if not hasattr(ary, "bhc_ary"):
        raise TypeError("must be a Bohrium array")
    raise NotImplementedError("TODO: data_np2bhc")
