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
import atexit
import numpy as np
import json
import os
from os.path import join
import _bh
import _info
import bhc

#Returns the Bohrium name of the data type of the Bohrium-C array
def dtype_from_bhc(bhc_ary):
    return  bhc_ary.__str__().rsplit("_",1)[-1]

#Returns the Bohrium name of the data type of the object 'obj'
#NB: use dtype_from_bhc() when 'obj' is a Bohrium-C array
def dtype_name(obj):
    if hasattr(obj, "dtype"):
        t = obj.dtype
    elif isinstance(obj, np.dtype):
        t = obj
    elif isinstance(obj, basestring):
        t = np.dtype(obj)
    else:
        t = np.dtype(type(obj))
    if t == np.bool_:
        return 'bool8'
    else:
        return t.name

#Returns the type signature (output, input) to use with the given operation.
#NB: we only returns the type of the first input thus all input types must be identical
def type_sig(op_name, inputs):
    f = _info.op[op_name]
    dtype = np.result_type(*inputs).name
    for sig in f['type_sig']:
        if dtype == sig[1]:
            return (np.dtype(sig[0]),np.dtype(sig[1]))
    raise TypeError("Cannot detement the correct signature (%s:%s)"%(op_name,dtype))

#Returns the Bohrium-C array
def get_bhc(ary):
    #Find the base array
    if ary.base is None:
        base = ary
    else:
        base = ary.base
        while base.base is not None:
            base = base.base

    #Create the base array in Bohrium
    if base.bhc_ary is None:
        if not ary.flags['BEHAVED']:
            raise ValueError("Bohrium arrays must be aligned, writeable, and in machine byte-order")
        if not ary.flags['OWNDATA']:
            raise ValueError("Bohrium base arrays must own its data")
        if not ary.flags['C_CONTIGUOUS']:
            raise ValueError("For now Bohrium only supports C-style arrays")

        #Lets create the new base array
        dtype = dtype_name(ary)
        size = ary.size
        print "ret = bhc.bh_multi_array_%s_new_empty(1, (%d,))"%(dtype,size)
        exec "ret = bhc.bh_multi_array_%s_new_empty(1, (%d,))"%(dtype,size)
        base.bhc_ary = ret

    if ary is base:#We a returning a base array
        return base.bhc_ary

bhc_arys_to_destroy = []
@atexit.register
def goodbye():
    for a in bhc_arys_to_destroy:
        print "bhc.bh_multi_array_%s_destroy(a)"%dtype_from_bhc(a)
        exec "bhc.bh_multi_array_%s_destroy(a)"%dtype_from_bhc(a)
