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
import numpy as np
import _info
import bhc

#Flush the delayed operations for Bohrium execution
def flush():
    bhc.bh_runtime_flush()

#Returns the Bohrium name of the data type of the Bohrium-C array
def dtype_from_bhc(bhc_ary):
    return  bhc_ary.__str__().rsplit("_",1)[-1]

#Returns the Bohrium name of the data type of the object 'obj'
#NB: use dtype_from_bhc() when 'obj' is a Bohrium-C array
def dtype_name(obj):
    if isinstance(obj, np.dtype):
        t = obj
    elif isinstance(obj, basestring):
        t = np.dtype(obj)
    elif isinstance(obj, type):
        t = np.dtype(obj)
    elif hasattr(obj, "dtype"):
        t = obj.dtype
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

