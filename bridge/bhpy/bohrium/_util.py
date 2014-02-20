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
import numpy
import json
import os
from os.path import join
import _bh
import bhc

#Returns the Bohrium name of the data type of the Bohrium-C array
def dtype_from_bhc(bhc_ary):
    return  bhc_ary.__str__().rsplit("_",1)[-1]

#Returns the Bohrium name of the data type of the object 'obj'
#NB: use dtype_from_bhc() when 'obj' is a Bohrium-C array
def dtype_name(obj):
    if isinstance(obj, numpy.ndarray):
        t = obj.dtype
    else:
        t = numpy.dtype(type(obj))
    if t == 'bool':
        t = 'bool8'
    return t

#Return a new bhc array based on 'numpy_array'
#NB: we always creates a flat array
def create_bhc_array(numpy_array):
    dtype = dtype_name(numpy_array)
    size = numpy_array.size
    print "ret = bhc.bh_multi_array_%s_new_empty(1, (%d,))"%(dtype,size)
    exec "ret = bhc.bh_multi_array_%s_new_empty(1, (%d,))"%(dtype,size)
    return ret

bhc_arys_to_destroy = []
@atexit.register
def goodbye():
    for a in bhc_arys_to_destroy:
        print "bhc.bh_multi_array_%s_destroy(a)"%dtype_from_bhc(a)
        exec "bhc.bh_multi_array_%s_destroy(a)"%dtype_from_bhc(a)
