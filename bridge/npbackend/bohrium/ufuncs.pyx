#!/usr/bin/env python
"""
=========================
NumPy ufunc encapsulation
=========================
"""

from __future__ import print_function
import sys
import os
import warnings
from . import _util
from . import array_create
import numpy_force as np
from . import _info
from ._util import dtype_equal
from .bhary import get_bhc, get_base, fix_biclass_wrapper, get_cdata
from . import bhary
from . import target
from .array_manipulation import broadcast_arrays, flatten

@fix_biclass_wrapper
def extmethod(name, out, in1, in2):
    # We need this, or else we need every combination of types in the opcodes.json
    assert in1.dtype == in2.dtype
    target.extmethod(name, get_bhc(out), get_bhc(in1), get_bhc(in2))

def setitem(ary, loc, value):
    """
    Set the 'value' into 'ary' at the location specified through 'loc'.
    'loc' can be a scalar or a slice object, or a tuple thereof
    """

    if not isinstance(loc, tuple):
        loc = (loc,)

    # Let's try to convert non-arrays and non-scalars to an array
    # e.g. converting a python list to an array
    if not (bhary.check(value) or np.isscalar(value)):
        value = array_create.array(value)

    # Lets make sure that not all dimensions are indexed by integers
    loc = list(loc)
    if len(loc) == ary.ndim and all((np.isscalar(s) for s in loc)):
        # 'slice' doesn't support negative start index
        if loc[0] < 0:
            loc[0] += ary.shape[0]
        loc[0] = slice(loc[0], loc[0] + 1)

    # Copy the 'value' to 'ary' using the 'loc'
    if ary.ndim == 0:
        assign(value, ary)
    else:
        assign(value, ary[tuple(loc)])

def overlap_conflict(out, *inputs):
    """
    Return True when there is a possible memory conflict between the output
    and the inputs.


    :param Mixed out: Array in the role of being written to.
    :param Mixed inputs: Arrays in the role being read from.
    :returns: True in case of conflict.
    :rtype: bool
    """

    for i in inputs:
        if not np.isscalar(i):
            if np.may_share_memory(out, i) and not (out.ndim == i.ndim and \
                                                                out.strides == i.strides and out.shape == i.shape and \
                                                                get_cdata(out) == get_cdata(i)):
                return True
    return False

@fix_biclass_wrapper
def assign(ary, out):
    """Copy data from array 'ary' to 'out'"""

    if not np.isscalar(ary):
        (ary, out) = broadcast_arrays(ary, out)[0]
        # We ignore self assignments
        if bhary.get_base(ary) is bhary.get_base(out) and \
                bhary.identical_views(ary, out):
            return

    # Assigning empty arrays doesn't do anything
    if hasattr(ary, "size"):
        if ary.size == 0:
            return
    if hasattr(out, "size"):
        if out.size == 0:
            return

    # We use a tmp array if the in-/out-put has memory conflicts
    if overlap_conflict(out, ary):
        tmp = array_create.empty_like(out)
        assign(ary, tmp)
        return assign(tmp, out)

    if bhary.check(out):
        out = get_bhc(out)
        if not np.isscalar(ary):
            if not bhary.check(ary):
                # Convert the NumPy array to bohrium
                ary = array_create.array(ary)
            ary = get_bhc(ary)
        target.ufunc(UFUNCS["identity"], out, ary)
    else:
        if bhary.check(ary):
            if "BH_SYNC_WARN" in os.environ:
                import warnings
                warnings.warn("BH_SYNC_WARN: Copying the array to NumPy", RuntimeWarning, stacklevel=2)
            get_base(ary)._data_bhc2np()
        out[...] = ary


class Ufunc(object):
    def __init__(self, info):
        """A Bohrium Universal Function"""
        self.info = info
        if sys.version_info.major >= 3:
            self.__name__ = info['name']
        else:
            # Scipy complains if '__name__' is unicode
            self.__name__ = info['name'].encode('latin_1')

    def __str__(self):
        return "<bohrium Ufunc '%s'>" % self.info['name']

    @fix_biclass_wrapper
    def __call__(self, *args, **kwargs):
        args = list(args)

        # Check number of array arguments
        if len(args) != self.info['nop'] and len(args) != self.info['nop'] - 1:
            raise ValueError("invalid number of array arguments")

        # Let's make sure that 'out' is always a positional argument
        try:
            out = kwargs['out']
            del kwargs['out']
            if len(args) == self.info['nop']:
                raise ValueError("cannot specify 'out' as both a positional and keyword argument")
            args.append(out)
        except KeyError:
            pass

        # We do not support NumPy's exotic arguments
        for k, val in kwargs.items():
            if val is not None:
                raise ValueError("Bohrium ufuncs doesn't support the '%s' argument" % str(k))

        # Broadcast the args
        (bargs, out_shape) = broadcast_arrays(*args)

        # Pop the output from the 'bargs' list
        out = None
        if len(args) == self.info['nop']:
            out = args.pop()
            if out_shape != out.shape:
                raise ValueError("non-broadcastable output operand with shape %s "
                                 "doesn't match the broadcast shape %s" %
                                 (str(args[-1].shape), str(out.shape)))

        # We use a tmp array if the in-/out-put has memory conflicts
        if out is not None:
            if overlap_conflict(out, *args):
                tmp = self.__call__(*args, **kwargs)
                assign(tmp, out)
                return out

        # Copy broadcasted array back to 'args' excluding scalars
        for i in range(len(args)):
            if not np.isscalar(args[i]):
                args[i] = bargs[i]

        if any([bhary.check(a) for a in args]):
            if out is not None and not bhary.check(out):
                raise NotImplementedError("For now, the output must be a Bohrium " \
                                          "array when the input arrays are")
        elif not bhary.check(out):
            # All operands are regular NumPy arrays
            func = eval("np.%s" % self.info['name'])
            if out is not None:
                args.append(out)
            return func(*args)

        if len(args) > 2:
            raise ValueError("Bohrium do not support ufunc with more than two inputs")

        # Find the type signature
        (out_dtype, in_dtype) = _util.type_sig(self.info['name'], args)

        # Convert dtype of all inputs
        for i in range(len(args)):
            if not np.isscalar(args[i]) and not dtype_equal(args[i], in_dtype):
                tmp = array_create.empty_like(args[i], dtype=in_dtype)
                tmp[...] = args[i]
                args[i] = tmp

        # Insert the output array
        if out is None or not dtype_equal(out_dtype, out.dtype):
            args.insert(0, array_create.empty(out_shape, out_dtype))
        else:
            args.insert(0, out)

        # Convert 'args' to Bohrium-C arrays
        bhcs = []
        for arg in args:
            if np.isscalar(arg):
                bhcs.append(arg)
            elif bhary.check(arg):
                bhcs.append(get_bhc(arg))
            else:
                arg = array_create.array(arg)
                bhcs.append(get_bhc(arg))

        # Some simple optimizations
        if self.info['name'] == "power" and np.isscalar(bhcs[2]) and bhcs[2] == 2:
            # Replace power of 2 with a multiplication
            target.ufunc(UFUNCS["multiply"], bhcs[0], bhcs[1], bhcs[1])
        else:
            target.ufunc(self, *bhcs)

        if out is None or dtype_equal(out_dtype, out.dtype):
            return args[0]
        else:
            # We need to convert the output type before returning
            assign(args[0], out)
            return out

    @fix_biclass_wrapper
    def reduce(self, ary, axis=0, out=None):
        """
        A Bohrium Reduction
        Reduces `ary`'s dimension by len('axis'), by applying ufunc along the
        axes in 'axis'.

        Let :math:`ary.shape = (N_0, ..., N_i, ..., N_{M-1})`.  Then
        :math:`ufunc.reduce(ary, axis=i)[k_0, ..,k_{i-1}, k_{i+1}, .., k_{M-1}]` =
        the result of iterating `j` over :math:`range(N_i)`, cumulatively applying
        ufunc to each :math:`ary[k_0, ..,k_{i-1}, j, k_{i+1}, .., k_{M-1}]`.
        For a one-dimensional array, reduce produces results equivalent to:

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
            if bhary.check(out):
                if not bhary.check(ary):
                    ary = array_create.array(ary)
            else:
                if bhary.check(ary):
                    ary = ary.copy2numpy()

        # Let NumPy handle NumPy array reductions
        if not bhary.check(ary):
            func = eval("np.%s.reduce" % self.info['name'])
            return func(ary, axis=axis, out=out)

        # Make sure that 'axis' is a list of dimensions to reduce
        if axis is None:
            # We reduce all dimensions
            axis = range(ary.ndim)
        elif np.isscalar(axis):
            # We reduce one dimension
            axis = [axis]
        else:
            # We reduce multiple dimensions
            axis = list(axis)

        # Check for out of bounds and convert negative axis values
        if len(axis) > ary.ndim:
            raise ValueError("number of 'axes' to reduce is out of bounds")
        for i in range(len(axis)):
            if axis[i] < 0:
                axis[i] = ary.ndim + axis[i]
            if axis[i] >= ary.ndim:
                raise ValueError("'axis' is out of bounds")

        # No axis should be reduced multiple times
        if len(axis) != len(set(axis)):
            raise ValueError("duplicate value in 'axis'")

        # TODO: is this a good idea?
        axis = sorted(axis, reverse=True)

        # When reducing booleans numerically, we count the number of True values
        if (not self.info['name'].startswith("logical")) and dtype_equal(ary, np.bool):
            ary = array_create.array(ary, dtype=np.uint64)

        if len(axis) == 1:
            # One axis reduction we can handle directly
            axis = axis[0]

            # Find the output shape
            if ary.ndim == 1:
                shape = []
            else:
                shape = tuple(s for i, s in enumerate(ary.shape) if i != axis)
                if out is not None and out.shape != shape:
                    raise ValueError("output dimension mismatch expect " \
                                     "shape '%s' got '%s'" % (shape, out.shape))

            tmp = array_create.empty(shape, dtype=ary.dtype)

            # NumPy compatibility: when the axis dimension size is zero NumPy just returns the neutral value
            if ary.shape[axis] == 0:
                tmp[...] = getattr(getattr(np, self.info['name']), "identity")
            else:
                target.reduce(self, get_bhc(tmp), get_bhc(ary), axis)

            if out is not None:
                out[...] = tmp
            else:
                out = tmp

            return out
        else:
            # Let's reduce the first axis
            ary = self.reduce(ary, axis[0])
            # Then we reduce the rest of the axes
            axis = axis[1:]
            ary = self.reduce(ary, axis)
            # Finally, we may have to copy the result to 'out'
            if out is not None:
                out[...] = ary
            else:
                out = ary
            return out

    @fix_biclass_wrapper
    def accumulate(self, ary, axis=0, out=None):
        """
        accumulate(ary, axis=0, out=None)

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
        ary : array_like
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
            if bhary.check(out):
                if not bhary.check(ary):
                    ary = array_create.array(ary)
            else:
                if bhary.check(ary):
                    ary = ary.copy2numpy()
            if out.shape != ary.shape:
                raise ValueError("output dimension mismatch expect " \
                                 "shape '%s' got '%s'" % (ary.shape, out.shape))

        # Check for out of bounds and convert negative axis values
        if axis < 0:
            axis = ary.ndim + axis
        if axis >= ary.ndim:
            raise ValueError("'axis' is out of bounds")

        # Let NumPy handle NumPy array accumulate
        if not bhary.check(ary):
            func = eval("np.%s.accumulate" % self.info['name'])
            return func(ary, axis=axis, out=out)

        # When reducing booleans numerically, we count the number of True values
        if dtype_equal(ary, np.bool):
            ary = array_create.array(ary, dtype=np.uint64)

        if out is None:
            out = array_create.empty(ary.shape, dtype=ary.dtype)

        target.accumulate(self, get_bhc(out), get_bhc(ary), axis)
        return out


#
# Expose all ufuncs at the module-level.
#
# After the following is executed, all ufuncs will be available as
# object instances of the Ufunc class via the dict of all ufuncs (UFUNCS)
# and via their individual names such as "negative", "identity", "add" etc.

# Expose via UFUNCS
UFUNCS = {}
for op in _info.op.values():
    f = Ufunc(op)
    UFUNCS[f.info['name']] = f

# 'bh_divide' refers to how Bohrium divide, which is like division in C/C++
# We needs this reference because Python v3 uses "true" division
UFUNCS["bh_divide"] = UFUNCS["divide"]


# NOTE: We have to add ufuncs that doesn't map to Bohrium operations directly
#       such as "negative" which can be done like below.
class Negative(Ufunc):
    @fix_biclass_wrapper
    def __call__(self, ary, out=None):
        if out is None:
            return -1 * ary
        else:
            out[...] = -1 * ary
            return out


UFUNCS["negative"] = Negative({'name': 'negative'})


class TrueDivide(Ufunc):
    @fix_biclass_wrapper
    def __call__(self, a1, a2, out=None):
        if _util.dtype_is_float(a1) or _util.dtype_is_float(a2):
            # Floating points automatically use Bohrium division
            ret = UFUNCS["bh_divide"](a1, a2)
        else:
            if a1.dtype.itemsize > 4 or a2.dtype.itemsize > 4:
                dtype = np.float64
            else:
                dtype = np.float32
            ret = array_create.array(a1, dtype=dtype) / array_create.array(a2, dtype=dtype)
        if out is None:
            return ret
        else:
            out[...] = ret
            return out


UFUNCS["true_divide"] = TrueDivide({'name': 'true_divide'})


class FloorDivide(Ufunc):
    @fix_biclass_wrapper
    def __call__(self, a1, a2, out=None):
        if _util.dtype_is_float(a1) or _util.dtype_is_float(a2):
            ret = UFUNCS["floor"](a1 / a2)
        else:
            # Integers automatically use Bohrium division
            ret = UFUNCS["bh_divide"](a1, a2)
        if out is None:
            return ret
        else:
            out[...] = ret
            return out


UFUNCS["floor_divide"] = FloorDivide({'name': 'floor_divide'})

# Python v3 uses "true" division
if sys.version_info.major >= 3:
    UFUNCS["divide"] = UFUNCS["true_divide"]

# Expose via their name.
for name, ufunc in UFUNCS.items():
    exec ("%s = ufunc" % name)

# We do not want to expose a function named "ufunc"
del ufunc


def _handle__array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    """
    - *self* is the array the method `__array_ufunc__` was called from.
    - *ufunc* is the ufunc object that was called.
    - *method* is a string indicating how the Ufunc was called, either
      ``"__call__"`` to indicate it was called directly, or one of its
      :ref:`methods<ufuncs.methods>`: ``"reduce"``, ``"accumulate"``,
      ``"reduceat"``, ``"outer"``, or ``"at"``.
    - *inputs* is a tuple of the input arguments to the ``ufunc``
    - *kwargs* contains any optional or keyword arguments passed to the
      function. This includes any ``out`` arguments, which are always
      contained in a tuple.
    """

    if method == '__call__' and ufunc.__name__ in UFUNCS:
        return UFUNCS[ufunc.__name__](*inputs, **kwargs)
    else:
        warnings.warn("Bohrium does not support ufunc `%s` it will be handled by "
                      "the original NumPy." % ufunc.__name__, UserWarning, 1)
        np_inputs = []
        for i in inputs:
            if bhary.check(i):
                np_inputs.append(i.copy2numpy())
            else:
                np_inputs.append(i)
        return getattr(np, ufunc.__name__)(*np_inputs, **kwargs)
