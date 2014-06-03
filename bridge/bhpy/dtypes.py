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
#import bhc
import numpy
import array_create
import ufunc

### For now, we do not need specialized Bohrium dtypes thus this file is not used ###

# This module consist of bohrium.ndarray data types

class float64(numpy.float64):
    npy_obj = numpy.float64
    name = "float64"
    def __str__(self):
        return "<bohrium type '%s'>"%self.name

    def __add__(self, rhs):
        print "my add"
        return ufunc.add(self, rhs)

dtypes = {'float64' :float64}

def dtype(obj, copy=False):
    """
    Create a data type object supported by Bohrium.

    A numpy array is homogeneous, and contains elements described by a
    dtype object. A dtype object can only be fundamental numeric types.

    Parameters
    ----------
    obj
        Object to be converted to a data type object.
    copy : bool, optional
        Make a new copy of the data-type object. If ``False``, the result
        may just be a reference to a built-in data-type object.

    See also
    --------
    result_type

    Examples
    --------
    Using array-scalar type:

    >>> np.dtype(np.int16)
    dtype('int16')
    """
    name = numpy.dtype(obj, copy=False).__str__()
    ret = dtypes.get(name, None)
    if ret is None:
        raise ValueError("The dtype '%s' is not supported by Bohrium"%name)
    return ret

d = float64
d1 = dtype("float64")
print d

a = array_create.ones(10,dtype=d)
print a.dtype
d(42) + a


