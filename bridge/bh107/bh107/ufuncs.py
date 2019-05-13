# -*- coding: utf-8 -*-
import sys
import numbers
import numpy as np
from bohrium_api import _bh_api, _info
from . import bharray, _dtype_util


class InvalidArgumentError(Exception):
    pass


def _result_dtype(op_name, inputs):
    """
    Returns the type signature (output, input) to use with the given operation.
    NB: we only returns the type of the first input thus all input types must
        be identical
    """

    func = _info.op[op_name]
    # Note that we first use the dtype before the array as inputs to result_type()
    inputs = [getattr(t, 'dtype', t) for t in inputs]
    dtype = np.result_type(*inputs)
    for sig in func['type_sig']:
        if dtype.name == sig[1]:
            return (_dtype_util.type_to_dtype(sig[0]), _dtype_util.type_to_dtype(dtype))

    # Let's try use a float signature for the integer input
    if np.issubdtype(dtype, np.integer):
        for sig in func['type_sig']:
            if 'float' in sig[1]:
                return (_dtype_util.type_to_dtype(sig[0]), _dtype_util.type_to_dtype(sig[1]))

    raise TypeError("The ufunc %s() does not support input data type: %s." % (op_name, dtype.name))


def _result_shape(shape_list):
    """Return the result of broadcasting `shapes` against each other"""

    # Find the number of dimensions of the broadcasted shape
    ret_ndim = 0
    for shape in shape_list:
        if len(shape) > ret_ndim:
            ret_ndim = len(shape)

    # Make sure that all shapes has the same length by pre-pending ones
    for i in range(len(shape_list)):
        shape_list[i] = [1] * (ret_ndim - len(shape_list[i])) + list(shape_list[i])

    # The resulting shape is the max of each dimension
    ret = []
    for i in range(ret_ndim):
        greatest = 0
        for shape in shape_list:
            if shape[i] > greatest:
                greatest = shape[i]
        ret.append(greatest)
    return tuple(ret)


def broadcast_to(ary, shape):
    """
    /** Return a new view of `ary` that is broadcasted to `shape`
     *  We use the term broadcast as defined by NumPy. Let `ret` be the broadcasted view of `ary`:
     *    1) One-sized dimensions are prepended to `ret.shape()` until it has the same number of dimension as `ary`.
     *    2) The stride of each one-sized dimension in `ret` is set to zero.
     *    3) The shape of `ary` is set to `shape`
     *
     *  \note See: <https://docs.scipy.org/doc/numpy-1.15.0/user/basics.broadcasting.html>
     *
     * @param ary    Input array
     * @param shape  The new shape
     * @return       The broadcasted array
     */
    """

    if len(ary.shape) > len(shape):
        raise InvalidArgumentError(
            "When broadcasting, the number of dimension of array cannot be greater than in the new shape")

    # Prepend ones to shape and zeros to stride in order to make them the same lengths as `shape`
    ret_shape = [1] * (len(shape) - len(ary.shape)) + ary.shape
    ret_strides = [0] * (len(shape) - len(ary.shape)) + ary.strides

    # Broadcast each dimension by setting ret_stride to zero and ret_shape to `shape`
    for i in range(len(ret_shape)):
        if ret_shape[i] != shape[i]:
            if ret_shape[i] == 1:
                ret_shape[i] = shape[i]
                ret_stride[i] = 0
            else:
                raise InvalidArgumentError("Cannot broadcast shape %s to %s" % (ary.shape, shape))
    return bharray.BhArray(ret_shape, ary.dtype, strides=ret_strides, offset=ary.offset, base=ary.base)


def _call_bh_api_op(op_id, out_operand, in_operand_list, broadcast_to_output_shape=True):
    dtype_enum_list = [_dtype_util.np2bh_enum(out_operand.dtype)]
    handle_list = [out_operand._bhc_handle]
    for op in in_operand_list:
        if isinstance(op, numbers.Number):
            dtype_enum_list.append(_dtype_util.np2bh_enum(_dtype_util.type_to_dtype(type(op))))
            if isinstance(op, (int, float, complex)):
                handle_list.append(op)
            elif isinstance(op, bool):
                handle_list.append(int(op))
            elif np.issubdtype(op, np.integer):
                handle_list.append(int(op))
            elif np.issubdtype(op, np.floating):
                handle_list.append(float(op))
            elif np.issubdtype(op, np.complex):
                handle_list.append(complex(op))
            else:
                raise InvalidArgumentError("NumPy scalar type must be an integer, float, or complex")
        else:
            dtype_enum_list.append(_dtype_util.np2bh_enum(out_operand.dtype))
            assert (op._bhc_handle is not None)
            if op.shape != out_operand.shape and broadcast_to_output_shape:
                op = broadcast_to(op, out_operand.shape)
            handle_list.append(op._bhc_handle)
    _bh_api.op(op_id, dtype_enum_list, handle_list)


def is_same_view(a, b):
    """ Return True when a and b is the same view. Their bases and dtypes might differ"""
    return a.offset == b.offset and a.shape == b.shape and a.strides == b.strides


def overlap_conflict(out, inputs):
    """ Return True when there is a possible memory conflict between the output and the inputs."""

    for i in inputs:
        # Scalars, different bases, or identical views can never conflict
        if not np.isscalar(i) and i.base is out.base and not is_same_view(out, i):
            o_low = out.offset
            i_low = i.offset
            o_high = o_low + 1
            i_high = i_low + 1
            for o_shape, o_stride, i_shape, i_stride in zip(out.shape, out.strides, i.shape, i.strides):
                if o_stride < 0:
                    o_low += (o_shape - 1) * o_stride
                else:
                    o_high += (o_shape - 1) * o_stride
                if i_stride < 0:
                    i_low += (i_shape - 1) * i_stride
                else:
                    i_high += (i_shape - 1) * i_stride

            if not (i_low >= o_high or o_low >= i_high):
                return True
    return False


def assign(src, dst):
    if dst.nelem > 0:
        if overlap_conflict(dst, [src]):  # We use a tmp array if the in-/out-put has memory conflicts
            tmp = bharray.BhArray(dst.shape, dst.dtype)
            _call_bh_api_op(_info.op["identity"]["id"], tmp, [src])
            _call_bh_api_op(_info.op["identity"]["id"], dst, [tmp])
        else:
            _call_bh_api_op(_info.op["identity"]["id"], dst, [src])


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

    def __call__(self, *operand_list):
        if len(operand_list) == self.info['nop']:
            out_operand = operand_list[-1]
            in_operands = list(operand_list[:-1])
            if not isinstance(out_operand, bharray.BhArray):
                raise InvalidArgumentError("Output must be of type `BhArray` is `%s`" % type(out_operand))
        elif len(operand_list) == self.info['nop'] - 1:
            out_operand = None
            in_operands = list(operand_list)
        else:
            raise InvalidArgumentError("The ufunc `%s` takes %d input arguments followed by an "
                                       "optional output argument" % (self.info['name'], self.info['nop'] - 1))

        out_shape = _result_shape([getattr(x, 'shape', (1,)) for x in operand_list])
        if out_operand is not None and out_shape != out_operand.shape:
            raise InvalidArgumentError(
                "Shape mismatch, the output shape %s should have been %s" % (out_operand.shape, out_shape))

        out_dtype, in_dtype = _result_dtype(self.info['name'], in_operands)

        # Convert dtype of all inputs to match the function type signature
        for i in range(len(in_operands)):
            if np.isscalar(in_operands[i]):
                if _dtype_util.type_to_dtype(type(in_operands[i])) != in_dtype:
                    in_operands[i] = in_dtype(in_operands[i])
            else:
                in_operands[i] = in_operands[i].astype(in_dtype, always_copy=False)

        # If the output is specified, its shape must match `out_shape`
        if out_operand is None:
            out_operand = bharray.BhArray(out_shape, out_dtype)
        elif out_operand.shape != out_shape:
            raise InvalidArgumentError("The output argument should have the shape: %s" % out_shape)

        if out_operand.nelem > 0:
            if out_dtype == out_operand.dtype and not overlap_conflict(out_operand, in_operands):
                _call_bh_api_op(self.info["id"], out_operand, in_operands)
            else:  # We use a tmp array if the in-/out-put has memory conflicts or different dtypes
                tmp_out = bharray.BhArray(out_shape, out_dtype)
                _call_bh_api_op(self.info["id"], tmp_out, in_operands)
                assign(tmp_out, out_operand)
        return out_operand

    def reduce(self, ary, axis=0, out=None):
        """Reduces `ary`'s dimension by len('axis'), by applying ufunc along the
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
        ary : BhArray
            The array to act on.
        axis : None or int or tuple of ints, optional
            Axis or axes along which a reduction is performed.
            The default (`axis` = 0) is perform a reduction over the first
            dimension of the input array. `axis` may be negative, in
            which case it counts from the last to the first axis.
            If this is `None`, a reduction is performed over all the axes.
            If this is a tuple of ints, a reduction is performed on multiple
            axes, instead of a single axis or all the axes as before.
        out : ndarray, optional
            A location into which the result is stored. If not provided, a
            freshly-allocated array is returned.

        Returns
        -------
        r : BhArray      The reduced array. If `out` was supplied, `r` is a reference to it.

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

        # Make sure that 'axis' is a list of dimensions to reduce
        if axis is None:
            # We reduce all dimensions
            axis = range(len(ary.shape))
        elif np.isscalar(axis):
            # We reduce one dimension
            axis = [axis]
        else:
            # We reduce multiple dimensions
            axis = list(axis)

        # Check for out of bounds and convert negative axis values
        if len(axis) > len(ary.shape):
            raise ValueError("number of 'axes' to reduce is out of bounds")
        for i in range(len(axis)):
            if axis[i] < 0:
                axis[i] = len(ary.shape) + axis[i]
            if axis[i] >= len(ary.shape):
                raise ValueError("'axis' is out of bounds")

        # No axis should be reduced multiple times
        if len(axis) != len(set(axis)):
            raise ValueError("duplicate value in 'axis'")

        if not isinstance(ary, bharray.BhArray):
            raise InvalidArgumentError("Input must be of type `BhArray` is `%s`" % type(ary))

        if out is not None and not isinstance(out, bharray.BhArray):
            raise InvalidArgumentError("Output must be of type `BhArray` is `%s`" % type(out))

        # When reducing booleans numerically, we count the number of True values
        if (not self.info['name'].startswith("logical")) and ary.dtype == np.bool:
            ary = ary.astype(np.uint64)

        if len(axis) == 1:  # One axis reduction we can handle directly
            axis = axis[0]

            # Find the output shape
            if len(ary.shape) == 1:
                shape = []
            else:
                shape = tuple(s for i, s in enumerate(ary.shape) if i != axis)
                if out is not None and out.shape != shape:
                    raise ValueError("output dimension mismatch expect "
                                     "shape '%s' got '%s'" % (shape, out.shape))

            tmp = bharray.BhArray(shape, ary.dtype, is_scalar=len(shape) == 0)

            # NumPy compatibility: when the axis dimension size is zero NumPy just returns the neutral value
            if ary.shape[axis] == 0:
                tmp[...] = getattr(getattr(np, self.info['name']), "identity")
            elif len(ary.shape) == 1 and ary.shape[0] == 1:  # Single element, no need to reduce
                tmp[...] = ary[0]
            elif ary.empty():
                tmp = ary
            else:
                _call_bh_api_op(_info.op["%s_reduce" % self.info['name']]['id'], tmp, [ary, np.int64(axis)],
                                broadcast_to_output_shape=False)
            if out is not None:
                out[...] = tmp
            else:
                out = tmp
            return out
        else:
            # If we are reducing to a scalar across several dimensions, reshape to a vector
            if len(ary.shape) == len(axis) and ary.iscontiguous():
                ary = ary.flatten(always_copy=False)
                ary = self.reduce(ary)
            else:
                # Let's reduce the last axis
                # TODO: Flatten as many inner dimensions as possible!
                ary = self.reduce(ary, axis[-1])
                ary = self.reduce(ary, axis[:-1])

            # Finally, we may have to copy the result to 'out'
            if out is not None:
                out[...] = ary
            else:
                out = ary
        return out

    def accumulate(self, ary, axis=0, out=None):
        """Accumulate the result of applying the operator to all elements.

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
        # Check for out of bounds and convert negative axis values
        if axis < 0:
            axis = ary.ndim + axis
        if axis >= ary.ndim:
            raise ValueError("'axis' is out of bounds")

        if not isinstance(ary, bharray.BhArray):
            raise InvalidArgumentError("Input must be of type `BhArray` is `%s`" % type(ary))

        # When accumulate booleans numerically, we count the number of True values
        if (not self.info['name'].startswith("logical")) and ary.dtype == np.bool:
            ary = ary.astype(np.uint64)

        if out is None:
            out = bharray.BhArray(shape=ary.shape, dtype=ary.dtype)
        else:
            if not isinstance(out, bharray.BhArray):
                raise InvalidArgumentError("Output must be of type `BhArray` is `%s`" % type(out))
        if ary.nelem > 0:
            _call_bh_api_op(_info.op["%s_accumulate" % self.info['name']]['id'], out, [ary, np.int64(axis)],
                            broadcast_to_output_shape=False)
        return out


def generate_ufuncs():
    ufuncs = {}
    for op in _info.op.values():
        if op['elementwise'] and op['name'] != 'identity':
            f = Ufunc(op)
            ufuncs[f.info['name']] = f

    # Bohrium divide is like division in C/C++ where floats are like
    # `true_divide` and integers are like `floor_divide` in NumPy
    ufuncs['bh_divide'] = ufuncs['divide']
    del ufuncs['divide']

    # NOTE: We have to add ufuncs that doesn't map to Bohrium operations directly
    #       such as "negative" which can be done like below.
    class Negative(Ufunc):
        def __call__(self, a, out=None):
            if out is None:
                return ufuncs['mul'](a, -1)
            else:
                return ufuncs['mul'](a, -1, out)

    ufuncs["negative"] = Negative({'name': 'negative', 'nop': 2})

    class TrueDivide(Ufunc):
        def __call__(self, a1, a2, out=None):
            if np.issubdtype(_dtype_util.obj_to_dtype(a1), np.inexact) or \
                    np.issubdtype(_dtype_util.obj_to_dtype(a2), np.inexact):
                ret = ufuncs["bh_divide"](a1, a2)
            else:
                if _dtype_util.size_of(_dtype_util.obj_to_dtype(a1)) > 4 or \
                        _dtype_util.size_of(_dtype_util.obj_to_dtype(a2)) > 4:
                    dtype = np.float64
                else:
                    dtype = np.float32
                if not np.isscalar(a1):
                    a1 = a1.astype(dtype)
                if not np.isscalar(a2):
                    a2 = a2.astype(dtype)
                ret = ufuncs['bh_divide'](a1, a2)
            if out is None:
                return ret
            else:
                assign(ret, out)
                return out

    ufuncs["true_divide"] = TrueDivide({'name': 'true_divide', 'nop': 3})
    ufuncs["divide"] = TrueDivide({'name': 'divide'})  # In NumPy, `divide` and `true_divide` is identical

    class FloorDivide(Ufunc):
        def __call__(self, a1, a2, out=None):
            if np.issubdtype(_dtype_util.obj_to_dtype(a1), np.inexact) or \
                    np.issubdtype(_dtype_util.obj_to_dtype(a2), np.inexact):
                ret = ufuncs['floor'](ufuncs["bh_divide"](a1, a2))
            else:
                ret = ufuncs['bh_divide'](a1, a2)
            if out is None:
                return ret
            else:
                assign(ret, out)
                return out

    ufuncs["floor_divide"] = FloorDivide({'name': 'floor_divide', 'nop': 3})
    return ufuncs


def generate_bh_operations():
    ret = {}
    for op in _info.op.values():
        f = Ufunc(op)
        ret[f.info['name']] = f
    return ret


# Generate all ufuncs
ufunc_dict = generate_ufuncs()
