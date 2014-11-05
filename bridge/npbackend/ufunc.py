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
from __future__ import print_function
from . import _util
from . import array_create
import numpy as np
from . import _info
from ._util import dtype_equal
from .ndarray import get_bhc, get_base, fix_returned_biclass
from . import ndarray
from . import target
from .array_manipulation import broadcast_arrays

@fix_returned_biclass
def extmethod(name, out, in1, in2):
    assert in1.dtype == in2.dtype
    target.extmethod(name, get_bhc(out), get_bhc(in1), get_bhc(in2))

def setitem(ary, loc, value):
    """Set the 'value' into 'ary' at the location specified through 'loc'.
    'loc' can be a scalar or a slice object, or a tuple thereof"""

    if not isinstance(loc, tuple):
        loc = (loc,)

    #Lets make sure that not all dimensions are indexed by integers
    loc = list(loc)
    if len(loc) == ary.ndim and all((np.isscalar(s) for s in loc)):
        if loc[0] < 0:#'slice' doesn't support negative start index
            loc[0] += ary.shape[0]
        loc[0] = slice(loc[0], loc[0]+1)
    #Copy the 'value' to 'ary' using the 'loc'
    assign(value, ary[tuple(loc)])

def overlap_conflict(out, *inputs):
    """Return True when there is a possible memory conflict between
    the output and the inputs"""

    for i in inputs:
        if not np.isscalar(i):
            if np.may_share_memory(out, i) and not (out.ndim == i.ndim and \
                     out.strides == i.strides and out.shape == i.shape and \
                     out.ctypes.data == i.ctypes.data):
                return True
    return False

@fix_returned_biclass
def assign(ary, out):
    """Copy data from array 'a' to 'out'"""

    if not np.isscalar(ary):
        (ary, out) = broadcast_arrays(ary, out)

    #We use a tmp array if the in-/out-put has memory conflicts
    if overlap_conflict(out, ary):
        tmp = array_create.empty_like(out)
        assign(ary, tmp)
        return assign(tmp, out)

    if ndarray.check(out):
        out = get_bhc(out)
        if not np.isscalar(ary):
            if not ndarray.check(ary):
                ary = array_create.array(ary)#Convert the NumPy array to bohrium
            ary = get_bhc(ary)
        target.ufunc(identity, out, ary)
    else:
        if ndarray.check(ary):
            get_base(ary)._data_bhc2np()
        out[...] = ary

class ufunc:
    def __init__(self, info):
        """A Bohrium Universal Function"""
        self.info = info
    def __str__(self):
        return "<bohrium ufunc '%s'>"%self.info['name']

    @fix_returned_biclass
    def __call__(self, *args, **kwargs):
        args = list(args)

        #Check number of array arguments
        if len(args) != self.info['nop'] and len(args) != self.info['nop']-1:
            raise ValueError("invalid number of array arguments")

        #Lets make sure that 'out' is always a positional argument
        try:
            out = kwargs['out']
            del kwargs['out']
            if len(args) == self.info['nop']:
                raise ValueError("cannot specify 'out' as both a positional and keyword argument")
            args.append(out)
        except KeyError:
            pass

        #We do not support NumPy's exotic arguments
        for k, val in kwargs.iteritems():
            if val is not None:
                raise ValueError(
                    "Bohrium funcs doesn't support the '%s' argument" % str(k)
                )

        #Broadcast the args
        bargs = broadcast_arrays(*args)

        #Pop the output from the 'bargs' list
        out = None
        if len(args) == self.info['nop']:#output given
            out = args.pop()
            if bargs[-1].shape != out.shape:
                raise ValueError("non-broadcastable output operand with shape %s "
                                 "doesn't match the broadcast shape %s"%
                                 (str(args[-1].shape), str(out.shape)))
        out_shape = bargs[-1].shape

        #We use a tmp array if the in-/out-put has memory conflicts
        if out is not None:
            if overlap_conflict(out, *args):
                tmp = self.__call__(*args, **kwargs)
                assign(tmp, out)
                return out

        #Copy broadcasted array back to 'args' excluding scalars
        for i in xrange(len(args)):
            if not np.isscalar(args[i]):
                args[i] = bargs[i]

        if any([ndarray.check(a) for a in args]):
            if out is not None and not ndarray.check(out):
                raise NotImplementedError("For now, the output must be a Bohrium "\
                                          "array when the input arrays are")
        elif not ndarray.check(out):#All operands are regular NumPy arrays
            func = eval("np.%s"%self.info['name'])
            if out is not None:
                args.append(out)
            return func(*args)

        if len(args) > 2:
            raise ValueError("Bohrium do not support ufunc with more than two inputs")

        #Find the type signature
        (out_dtype, in_dtype) = _util.type_sig(self.info['name'], args)

        #Convert dtype of all inputs
        for i in xrange(len(args)):
            if not np.isscalar(args[i]) and not dtype_equal(args[i], in_dtype):
                tmp = array_create.empty_like(args[i], dtype=in_dtype)
                tmp[...] = args[i]
                args[i] = tmp

        #Insert the output array
        if out is None or not dtype_equal(out_dtype, out.dtype):
            args.insert(0, array_create.empty(out_shape, out_dtype))
        else:
            args.insert(0, out)

        #Convert 'args' to Bohrium-C arrays
        bhcs = []
        for arg in args:
            if np.isscalar(arg):
                bhcs.append(arg)
            elif ndarray.check(arg):
                bhcs.append(get_bhc(arg))
            else:
                arg = array_create.array(arg)
                bhcs.append(get_bhc(arg))

        #Some simple optimizations
        if self.info['name'] == "power" and np.isscalar(bhcs[2]) and bhcs[2] == 2:
            #Replace power of 2 with a multiplication
            target.ufunc(multiply, bhcs[0], bhcs[1], bhcs[1])
        else:
            target.ufunc(self, *bhcs)

        if out is None or dtype_equal(out_dtype, out.dtype):
            return args[0]
        else:#We need to convert the output type before returning
            assign(args[0], out)
            return out
        return out

    @fix_returned_biclass
    def reduce(self, ary, axis=0, out=None):
        """
        A Bohrium Reduction
    Reduces `ary`'s dimension by one, by applying ufunc along one axis.

    Let :math:`ary.shape = (N_0, ..., N_i, ..., N_{M-1})`.  Then
    :math:`ufunc.reduce(ary, axis=i)[k_0, ..,k_{i-1}, k_{i+1}, .., k_{M-1}]` =
    the result of iterating `j` over :math:`range(N_i)`, cumulatively applying
    ufunc to each :math:`ary[k_0, ..,k_{i-1}, j, k_{i+1}, .., k_{M-1}]`.
    For a one-dimensional array, reduce produces results equivalent to:
    ::

     r = op.identity # op = ufunc
     for i in range(len(A)):
       r = op(r, A[i])
     return r

    For example, add.reduce() is equivalent to sum().

    Parameters
    ----------
    ary : array_like
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
    r : ndarraout      The reduced array. If `out` was supplied, `r` is a reference to it.

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

        if out is not None:
            if ndarray.check(out):
                if not ndarray.check(ary):
                    ary = array_create.array(ary)
            else:
                if ndarray.check(ary):
                    ary = a.copy2numpy()
        #Let NumPy handle NumPy array reductions
        if not ndarray.check(ary):
            func = eval("np.%s.reduce" % self.info['name'])
            return func(ary, axis=axis, out=out)

        #Make sure that 'axis' is a list of dimensions to reduce
        if axis is None:
            axis = range(ary.ndim)#We reduce all dimensions
        elif np.isscalar(axis):
            axis = [axis]#We reduce one dimension
        else:
            axis = list(axis)#We reduce multiple dimensions

        #When reducting booleans we count the number of True values
        if dtype_equal(ary, np.bool):
            ary = array_create.array(ary, dtype=np.uint64)

        #Check for out of bounds and convert negative axis values
        if len(axis) > ary.ndim:
            raise ValueError("number of 'axises' to reduce is out of bounds")
        for i in xrange(len(axis)):
            if axis[i] < 0:
                axis[i] = ary.ndim+axis[i]
            if axis[i] >= ary.ndim:
                raise ValueError("'axis' is out of bounds")

        if len(axis) == 1:#One axis reduction we can handle directly
            axis = axis[0]

            #Find the output shape
            if ary.ndim == 1:
                shape = (1,)
            else:
                shape = tuple(s for i, s in enumerate(ary.shape) if i != axis)
                if out is not None and out.shape != shape:
                    raise ValueError("output dimension mismatch expect "\
                                     "shape '%s' got '%s'"%(shape, out.shape))

            tmp = array_create.empty(shape, dtype=ary.dtype)
            target.reduce(self, get_bhc(tmp), get_bhc(ary), axis)

            if out is not None:
                out[...] = tmp
            else:
                out = tmp
            if ary.ndim == 1:#return a Python Scalar
                return out[0]
            else:
                return out
        else:
            tmp1 = self.reduce(ary, axis[0])
            axis = [i-1 for i in axis[1:]]
            tmp2 = self.reduce(tmp1, axis)
            if out is not None:
                out[...] = tmp2
            else:
                out = tmp2
            return out

    @fix_returned_biclass
    def accumulate(self, ary, axis=0, out=None):
        """
    accumulate(array, axis=0, out=None)

    Accumulate the result of applying the operator to all elements.

    For a one-dimensional array, accumulate produces results equivalent to::

      r = np.empty(len(A))
      t = op.identity        # op = the ufunc being applied to A's  elements
      for i in range(len(A)):
          t = op(t, A[i])
          r[i] = t
      return r

    For example, add.accumulate() is equivalent to np.cumsum().

    For a multi-dimensional array, accumulate is applied along only one
    axis (axis zero by default; see Examples below) so repeated use is
    necessary if one wants to accumulate over multiple axes.

    Parameters
    ----------
    array : array_like
        The array to act on.
    axis : int, optional
        The axis along which to apply the accumulation; default is zero.
    out : ndarray, optional
        A location into which the result is stored. If not provided a
        freshly-allocated array is returned.

    Returns
    -------
    r : ndarray
        The accumulated values. If `out` was supplied, `r` is a reference to
        `out`.

    Examples
    --------
    1-D array examples:

    >>> np.add.accumulate([2, 3, 5])
    array([ 2,  5, 10])
    >>> np.multiply.accumulate([2, 3, 5])
    array([ 2,  6, 30])

    2-D array examples:

    >>> I = np.eye(2)
    >>> I
    array([[ 1.,  0.],
           [ 0.,  1.]])

    Accumulate along axis 0 (rows), down columns:

    >>> np.add.accumulate(I, 0)
    array([[ 1.,  0.],
           [ 1.,  1.]])
    >>> np.add.accumulate(I) # no axis specified = axis zero
    array([[ 1.,  0.],
           [ 1.,  1.]])

    Accumulate along axis 1 (columns), through rows:

    >>> np.add.accumulate(I, 1)
    array([[ 1.,  1.],
           [ 0.,  1.]])
        """
        if out is not None:
            if ndarray.check(out):
                if not ndarray.check(ary):
                    ary = array_create.array(ary)
            else:
                if ndarray.check(ary):
                    ary = ary.copy2numpy()
            if out.shape != ary.shape:
                raise ValueError("output dimension mismatch expect "\
                                 "shape '%s' got '%s'"%(ary.shape, out.shape))

        #Let NumPy handle NumPy array accumulate
        if not ndarray.check(ary):
            func = eval("np.%s.accumulate" % self.info['name'])
            return func(ary, axis=axis, out=out)

        if out is None:
            out = array_create.empty(ary.shape, dtype=ary.dtype)

        target.accumulate(self, get_bhc(out), get_bhc(ary), axis)
        return out

#We have to add ufuncs that doesn't map to Bohrium operations directly
class negative(ufunc):
    def __call__(self, ary, out=None):
        if out is None:
            return -1 * ary
        else:
            out[...] = -1 * ary
            return out

#Expose all ufuncs
ufuncs = [negative({'name':'negative'})]
for op in _info.op.itervalues():
    ufuncs.append(ufunc(op))

for f in ufuncs:
    exec("%s = f" % f.info['name'])


###############################################################################
################################ UNIT TEST ####################################
###############################################################################

import unittest

class Tests(unittest.TestCase):

    def test_assign_copy(self):
        A = array_create.empty((4, 4), dtype=int)
        B = array_create.empty((4, 4), dtype=int)
        assign(42, A)
        assign(A, B)
        A = A.copy2numpy()
        B = B.copy2numpy()
        #Compare result to NumPy
        N = np.empty((4, 4), dtype=int)
        N[...] = 42
        self.assertTrue(np.array_equal(B, N))
        self.assertTrue(np.array_equal(A, N))

    def test_ufunc(self):
        for func in ufuncs:
            for type_sig in func.info['type_sig']:
                if func.info['name'] == "identity":
                    continue
                print(func, type_sig)
                A = array_create.empty((4, 4), dtype=type_sig[1])
                if type_sig[1] == "bool":
                    assign(False, A)
                else:
                    assign(2, A)
                if func.info['nop'] == 2:
                    res = func(A)
                elif func.info['nop'] == 3:
                    B = array_create.empty((4, 4), dtype=type_sig[2])
                    if type_sig[1] == "bool":
                        assign(True, B)
                    else:
                        assign(3, B)
                    res = func(A, B)
                res = res.copy2numpy()
                #Compare result to NumPy
                A = np.empty((4, 4), dtype=type_sig[1])
                A[...] = 2
                B = np.empty((4, 4), dtype=type_sig[1])
                B[...] = 3
                if func.info['nop'] == 2:
                    exec("np_res = np.%s(A)" % func.info['name'])
                elif func.info['nop'] == 3:
                    exec("np_res = np.%s(A,B)" % func.info['name'])
                self.assertTrue(np.allclose(res, np_res))

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Tests)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
