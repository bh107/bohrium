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
import _bh
import array_create
import bhc
import numpy as np
import _info

def assign(a, out):
    dtype = _util.dtype_name(out)
    if np.isscalar(a):
        cmd = "bhc.bh_multi_array_%s_assign_scalar(out.bhc_ary,a)"%(dtype)
    else:
        assert(dtype == _util.dtype_name(a))
        cmd = "bhc.bh_multi_array_%s_assign_array(out.bhc_ary,a.bhc_ary)"%(dtype)
    exec cmd
    return out

class ufunc:
    def __init__(self, info):
        """A Bohrium Universal Function"""
        self.info = info
    def __str__(self):
        return "<bohrium ufunc '%s'>"%self.info['bhc_name']
    def __call__(self, *args):

        #Check number of arguments
        if len(args) > self.info['nop'] or len(args) < self.info['nop']-1:
            raise ValueError("invalid number of arguments")

        #Find the data type of the input array
        dtype = None
        for a in args:
            if isinstance(a, _bh.ndarray):
                dtype = a.dtype
            else:
                raise NotImplementedError("For now we only supports Bohrium arrays")

        if dtype is None:
            raise ValueError("at least one of inputs must be a bohrium array")

        #Find the output array
        if len(args) < self.info['nop']:#No output given
            out = array_create.empty(dtype)
        else:
            out = args[-1]

        #Check for not implemented errors
        for a in args:
            if not isinstance(a, _bh.ndarray):
                raise NotImplementedError("All operands must be Bohrium arrays")
            if a.base is not None:
                raise NotImplementedError("We do not support views")

        f = eval("bhc.bh_multi_array_%s_%s"%(dtype, self.info['bhc_name']))

        if self.info['nop'] == 2:
            ret = f(args[0].bhc_ary)
        elif self.info['nop'] == 3:
            ret = f(args[0].bhc_ary, args[1].bhc_ary)

        #Copy result into the output array
        exec "bhc.bh_multi_array_%s_assign_array(out.bhc_ary,ret)"%(dtype)
        return ret

ufuncs = []
for op in _info.op.itervalues():
    ufuncs.append(ufunc(op))


###############################################################################
################################ UNIT TEST ####################################
###############################################################################

import unittest

class Tests(unittest.TestCase):

    def tes1t_assign_copy(self):
        A = array_create.empty((4,4), dtype=int)
        B = array_create.empty((4,4), dtype=int)
        assign(42, A)
        assign(A, B)

    def test_ufunc(self):
        for f in ufuncs:
            print f
            A = array_create.empty((4,4))
            assign(2, A)
            if f.info['nop'] == 2:
                res = f(A)
            elif f.info['nop'] == 3:
                B = array_create.empty((4,4))
                assign(3, B)
                res = f(A,B)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
    unittest.TextTestRunner(verbosity=2).run(suite)
