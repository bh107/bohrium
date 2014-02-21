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

        #Find the target data type
        dtype = np.result_type(args)

        #Find the output array
        out = None
        out_final = None #If not None the output needs type conversion
        if len(args) == self.info['nop']:#output given
            out = args.pop()
            if out.dtype != dtype:
                out_final = out
                out = None

        #Broadcast the inputs
        args = np.broadcast_arrays(args)

        #Create output array
        if out is None:
            out = array_create.empty(args[0].shape, dtype)

        if not np.array_equal(out.shape, args[0].shape):
            raise ValueError("the output and input shape doesn't match")

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
