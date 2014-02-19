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
import array_create
import bhc
import numpy as np

def assign(a, out=None):
    if out is None:
        out = array_create.empty(a.shape, a.dtype)

    if np.isscalar(a):
        cmd = "bhc.bh_multi_array_%s_assign_scalar(out.bhc_ary,a)"%(out.dtype.name)
        exec cmd
    return out

class ufunc:
    def __init__(self, info):
        """A Bohrium Universal Function"""
        self.info = info
    def __str__(self):
        return "<bohrium ufunc '%s'>"%self.info['npy']

class ufunc_binary(ufunc):
    """A Binary Bohrium Universal Function"""
    def __call__(self, X, Y, out=None):
        assert X.dtype == Y.dtype
        print "ret = bhc.bh_multi_array_%s_%s ("%(X.dtype.name,self.info['npy']),
        exec "ret = bhc.bh_multi_array_%s_%s(X.bhc_ary,Y.bhc_ary)"%(X.dtype.name,self.info['npy'])
        return ret

ufuncs = []
for op in _util.elementwise_opcodes:
    if op['nop'] == 2:
        pass
    elif op['nop'] == 3:
        f = ufunc_binary(op)
    ufuncs.append(f)



###############################################################################
################################ UNIT TEST ####################################
###############################################################################

import unittest

class Tests(unittest.TestCase):

    def test_assign(self):
        A = array_create.empty((4,4), dtype=np.float32)
        assign(np.float32(42), A)

    def t1est_ufunc(self):
        for f in ufuncs:
            print f
            A = array_create.empty((4,4))
            B = array_create.empty((4,4))
            print A.bhc_ary
            C = f(A,B)
            return




if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
    unittest.TextTestRunner(verbosity=2).run(suite)
