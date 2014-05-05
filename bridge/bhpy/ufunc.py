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
import _info
from _util import dtype_name, dtype_identical
from ndarray import get_bhc, check_biclass
import ndarray

def extmethod(name, out, in1, in2):
    assert in1.dtype == in2.dtype
    if check_biclass(out) or check_biclass(in1) or check_biclass(in2):
        raise NotImplementedError("NumPy views that points to Bohrium base "\
                                  "arrays isn't supported")

    f = eval("bhc.bh_multi_array_extmethod_%s_%s_%s"%(dtype_name(out),\
              dtype_name(in1), dtype_name(in2)))
    f(name, get_bhc(out), get_bhc(in1), get_bhc(in2))

def assign(a, out):
    if check_biclass(a) or check_biclass(out):
        raise NotImplementedError("NumPy views that points to Bohrium base "\
                                  "arrays isn't supported")
    if ndarray.check(out):
        np.broadcast(a,out)#We only do this for the dimension mismatch check
        out_dtype = dtype_name(out)
        out_bhc = get_bhc(out)
        if np.isscalar(a):
            exec "bhc.bh_multi_array_%s_assign_scalar(out_bhc,a)"%(out_dtype)
        else:
            if not ndarray.check(a):
                a = array_create.array(a)#Convert the NumPy array to bohrium
            a_bhc = get_bhc(a)
            a_dtype = dtype_name(a)
            if out_dtype != a_dtype:
                exec "a_bhc = bhc.bh_multi_array_%s_convert_%s(a_bhc)"%(out_dtype, a_dtype)
            exec "bhc.bh_multi_array_%s_assign_array(out_bhc,a_bhc)"%(out_dtype)
    elif ndarray.check(a):
        a._data_bhc2np()
        a.__array_priority__ = -1.0#Force NumPy to handle the assignment
        out[:] = a
        a.__array_priority__ = 2.0
    else:
        out[:] = a#Regular NumPy assignment

class ufunc:
    def __init__(self, info):
        """A Bohrium Universal Function"""
        self.info = info
    def __str__(self):
        return "<bohrium ufunc '%s'>"%self.info['name']
    def __call__(self, *args):

        #Check number of arguments
        if len(args) != self.info['nop'] and len(args) != self.info['nop']-1:
            raise ValueError("invalid number of arguments")

        for a in args:
            if check_biclass(a):
                raise NotImplementedError("NumPy views that points to Bohrium base "\
                                          "arrays isn't supported")

        #Check for shape mismatch and get the final output shape
        out_shape = np.broadcast(*args).shape if len(args) > 1 else args[0].shape

        #Pop the output from the 'args' list
        out = None
        args = list(args)
        if len(args) == self.info['nop']:#output given
            out = args.pop()
            if out.shape != out_shape:
                raise ValueError("Could not broadcast to the shape of the output array")

        if any([ndarray.check(a) for a in args]):
            if out is not None and not ndarray.check(out):
                raise NotImplementedError("For now, the output must be a Bohrium "\
                                          "array when the input arrays are")
        elif not ndarray.check(out):#All operands are regular NumPy arrays
            f = eval("np.%s"%self.info['name'])
            if out is not None:
                args.append(out)
            return f(*args)

        if len(args) > 2:
            raise ValueError("Bohrium do not support ufunc with more than two inputs")

        #Find the type signature
        (out_dtype,in_dtype) = _util.type_sig(self.info['name'], args)

        #Convert dtype of all inputs
        for i in xrange(len(args)):
            if not np.isscalar(args[i]) and not dtype_identical(args[i], in_dtype):
                t = array_create.empty_like(args[i], dtype=in_dtype)
                t[:] = args[i]
                args[i] = t;

        #Insert the output array
        if out is None or out_dtype != out.dtype:
            args.insert(0,array_create.empty(out_shape, out_dtype))
        else:
            args.insert(0,out)

        #Convert 'args' to Bohrium-C arrays
        bhcs = []
        tmps = []
        for a in args:
            if np.isscalar(a):
                bhcs.append(a)
            elif ndarray.check(a):
                bhcs.append(get_bhc(a))
            else:
                a = array_create.array(a)
                bhcs.append(get_bhc(a))
                tmps.append(a)#We use this to keep a reference to 'a'

        #Create and execute the ufunc command
        cmd = "bhc.bh_multi_array_%s_%s"%(dtype_name(in_dtype), self.info['name'])
        for i,a in enumerate(args):
            if np.isscalar(a):
                if i == 1:
                    cmd += "_scalar_lhs"
                if i == 2:
                    cmd += "_scalar_rhs"
        f = eval(cmd)
        f(*bhcs)

        del tmps#Now we can safely de-allocate the tmp input arrays

        if out is None or out_dtype == out.dtype:
            return args[0]
        else:#We need to convert the output type before returning
            f = eval("bhc.bh_multi_array_%s_convert_%s"%(dtype_name(args[0].dtype), dtype_name(out_dtype)))
            t = f(args[0])
            exec "bhc.bh_multi_array_%s_assign_array(get_bhc(out),t)"%(dtype_name(out_dtype))
            return out
        return out

    def reduce(self, a, axis=0, out=None):
        """ A Bohrium Reduction """

        if check_biclass(a):
            raise NotImplementedError("NumPy views that points to Bohrium base "\
                                      "arrays isn't supported")

        if not ndarray.check(a):#Let NumPy handle NumPy array reductions
            f = eval("np.%s.reduce"%self.info['name'])
            return f(a, axis=axis, out=None)

        if axis >= a.ndim:
            raise ValueError("'axis' is out of bound")
        if axis < 0:
            axis = a.ndim+axis

        if a.ndim == 1:
            shape = (1,)
        else:
            shape = tuple(s for i, s in enumerate(a.shape) if i != axis)
            if out is not None and out.shape != shape:
                raise ValueError("output dimension mismatch expect shape '%s' got '%s'"%(shape, out.shape))

        f = eval("bhc.bh_multi_array_%s_partial_reduce_%s"%(dtype_name(a), self.info['name']))
        ret = f(get_bhc(a),axis)
        t = ndarray.new(shape, a.dtype, ret)

        if a.ndim == 1:#return a Python Scalar
            return t[0]
        elif out is None:#return the new array output
            return t
        else:
            out[:] = t#Copy the new array to the given output array
            return out

#We have to add ufuncs that doesn't map to Bohrium operations directly
class negative(ufunc):
    def __call__(self, a, out=None):
        if out is None:
            return -1 * a
        else:
            out[:] = -1 * a
            return out

#Expose all ufuncs
ufuncs = [negative({'name':'negative'})]
for op in _info.op.itervalues():
    ufuncs.append(ufunc(op))

for f in ufuncs:
    exec "%s = f"%f.info['name']


###############################################################################
################################ UNIT TEST ####################################
###############################################################################

import unittest

class Tests(unittest.TestCase):

    def test_assign_copy(self):
        A = array_create.empty((4,4), dtype=int)
        B = array_create.empty((4,4), dtype=int)
        assign(42, A)
        assign(A, B)
        A = A.copy2numpy()
        B = B.copy2numpy()
        #Compare result to NumPy
        N = np.empty((4,4), dtype=int)
        N[:] = 42
        self.assertTrue(np.array_equal(B,N))
        self.assertTrue(np.array_equal(A,N))

    def test_ufunc(self):
        for f in ufuncs:
            for type_sig in f.info['type_sig']:
                if f.info['name'] == "identity":
                    continue
                print f, type_sig
                A = array_create.empty((4,4), dtype=type_sig[1])
                if type_sig[1] == "bool":
                    assign(False, A)
                else:
                    assign(2, A)
                if f.info['nop'] == 2:
                    res = f(A)
                elif f.info['nop'] == 3:
                    B = array_create.empty((4,4), dtype=type_sig[2])
                    if type_sig[1] == "bool":
                        assign(True, B)
                    else:
                        assign(3, B)
                    res = f(A,B)
                res = res.copy2numpy()
                #Compare result to NumPy
                A = np.empty((4,4), dtype=type_sig[1])
                A[:] = 2
                B = np.empty((4,4), dtype=type_sig[1])
                B[:] = 3
                if f.info['nop'] == 2:
                    exec "np_res = np.%s(A)"%f.info['name']
                elif f.info['nop'] == 3:
                    exec "np_res = np.%s(A,B)"%f.info['name']
                self.assertTrue(np.allclose(res,np_res))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
    unittest.TextTestRunner(verbosity=2).run(suite)
