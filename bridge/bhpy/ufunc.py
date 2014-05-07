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
from ndarray import get_bhc, check_biclass, del_bhc_obj
import ndarray

def extmethod(name, out, in1, in2):
    assert in1.dtype == in2.dtype
    if check_biclass(out) or check_biclass(in1) or check_biclass(in2):
        raise NotImplementedError("NumPy views that points to Bohrium base "\
                                  "arrays isn't supported")

    f = eval("bhc.bh_multi_array_extmethod_%s_%s_%s"%(dtype_name(out),\
              dtype_name(in1), dtype_name(in2)))
    bhc_out = get_bhc(out)
    bhc_in1 = get_bhc(in1)
    bhc_in2 = get_bhc(in2)
    f(name, bhc_out, bhc_in1, bhc_in2)
    del_bhc_obj(bhc_out)
    del_bhc_obj(bhc_in1)
    del_bhc_obj(bhc_in2)

def assign(a, out):
    if check_biclass(a) or check_biclass(out):
        raise NotImplementedError("NumPy views that points to Bohrium base "\
                                  "arrays isn't supported")
    if ndarray.check(out):
        np.broadcast(a,out)#We only do this for the dimension mismatch check
        out_dtype = dtype_name(out)
        out_bhc = get_bhc(out)
        a_dtype = dtype_name(a)
        cmd = "bhc.bh_multi_array_%s_identity_%s"%(out_dtype, a_dtype)
        if np.isscalar(a):
            exec "%s_scalar(out_bhc, a)"%cmd
        else:
            if not ndarray.check(a):
                a = array_create.array(a)#Convert the NumPy array to bohrium
            a_bhc = get_bhc(a)
            exec "%s(out_bhc, a_bhc)"%cmd
            del_bhc_obj(a_bhc)
        del_bhc_obj(out_bhc)
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

        #Cleanup
        for a in bhcs:
            if not np.isscalar(a):
                del_bhc_obj(a)
        del tmps#Now we can safely de-allocate the tmp input arrays

        if out is None or out_dtype == out.dtype:
            return args[0]
        else:#We need to convert the output type before returning
            f = eval("bhc.bh_multi_array_%s_convert_%s"%(dtype_name(args[0].dtype), dtype_name(out_dtype)))
            t = f(args[0])
            bhc_out = get_bhc(out)
            exec "bhc.bh_multi_array_%s_assign_array(bhc_out,t)"%(dtype_name(out_dtype))
            del_bhc_obj(bhc_out)
            return out
        return out

    def reduce(self, a, axis=0, out=None):
        """ A Bohrium Reduction
    Reduces `a`'s dimension by one, by applying ufunc along one axis.

    Let :math:`a.shape = (N_0, ..., N_i, ..., N_{M-1})`.  Then
    :math:`ufunc.reduce(a, axis=i)[k_0, ..,k_{i-1}, k_{i+1}, .., k_{M-1}]` =
    the result of iterating `j` over :math:`range(N_i)`, cumulatively applying
    ufunc to each :math:`a[k_0, ..,k_{i-1}, j, k_{i+1}, .., k_{M-1}]`.
    For a one-dimensional array, reduce produces results equivalent to:
    ::

     r = op.identity # op = ufunc
     for i in range(len(A)):
       r = op(r, A[i])
     return r

    For example, add.reduce() is equivalent to sum().

    Parameters
    ----------
    a : array_like
        The array to act on.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a reduction is performed.
        The default (`axis` = 0) is perform a reduction over the first
        dimension of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.

        .. versionadded:: 1.7.0

        If this is `None`, a reduction is performed over all the axes.
        If this is a tuple of ints, a reduction is performed on multiple
        axes, instead of a single axis or all the axes as before.

        For operations which are either not commutative or not associative,
        doing a reduction over multiple axes is not well-defined. The
        ufuncs do not currently raise an exception in this case, but will
        likely do so in the future.
    out : ndarray, optional
        A location into which the result is stored. If not provided, a
        freshly-allocated array is returned.

    Returns
    -------
    r : ndarray
        The reduced array. If `out` was supplied, `r` is a reference to it.

    Examples
    --------
    >>> np.multiply.reduce([2,3,5])
    30

    A multi-dimensional array example:

    >>> X = np.arange(8).reshape((2,2,2))
    >>> X
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.add.reduce(X, 0)
    array([[ 4,  6],
           [ 8, 10]])
    >>> np.add.reduce(X) # confirm: default axis value is 0
    array([[ 4,  6],
           [ 8, 10]])
    >>> np.add.reduce(X, 1)
    array([[ 2,  4],
           [10, 12]])
    >>> np.add.reduce(X, 2)
    array([[ 1,  5],
           [ 9, 13]])
        """

        if check_biclass(a):
            raise NotImplementedError("NumPy views that points to Bohrium base "\
                                      "arrays isn't supported")
        if out is not None:
            if ndarray.check(out):
                if not ndarray.check(a):
                    a = array_create.array(a)
            else:
                if ndarray.check(a):
                    a = a.copy2numpy()
        #Let NumPy handle NumPy array reductions
        if not ndarray.check(a):
            f = eval("np.%s.reduce"%self.info['name'])
            return f(a, axis=axis, out=None)

        #Make sure that 'axis' is a list of dimensions to reduce
        if axis is None:
            axis = range(a.ndim)#We reduce all dimensions
        elif np.isscalar(axis):
            axis = [axis]#We reduce one dimension
        else:
            axis = list(axis)#We reduce multiple dimensions

        #Check for out of bounds and convert negative axis values
        if len(axis) > a.ndim:
            raise ValueError("number of 'axises' to reduce is out of bounds")
        for i in xrange(len(axis)):
            if axis[i] < 0:
                axis[i] = a.ndim+axis[i]
            if axis[i] >= a.ndim:
                raise ValueError("'axis' is out of bounds")

        if len(axis) == 1:#One axis reduction we can handle directly
            axis = axis[0]

            #Find the output shape
            if a.ndim == 1:
                shape = (1,)
            else:
                shape = tuple(s for i, s in enumerate(a.shape) if i != axis)
                if out is not None and out.shape != shape:
                    raise ValueError("output dimension mismatch expect "\
                                     "shape '%s' got '%s'"%(shape, out.shape))

            f = eval("bhc.bh_multi_array_%s_%s_reduce"%(dtype_name(a), self.info['name']))
            tmp = array_create.empty(shape, dtype=a.dtype)
            tmp_bhc = get_bhc(tmp)
            a_bhc = get_bhc(a)
            f(tmp_bhc, a_bhc, axis)
            del_bhc_obj(tmp_bhc)
            del_bhc_obj(a_bhc)

            if out is not None:
                out[:] = tmp
            else:
                out = tmp
            if a.ndim == 1:#return a Python Scalar
                return out[0]
            else:
                return out
        else:
            t1 = self.reduce(a, axis[0])
            axis = [i-1 for i in axis[1:]]
            t2 = self.reduce(t1, axis)
            if out is not None:
                out[:] = t2
            else:
                out = t2
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
