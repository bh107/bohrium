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

    if ary.ndim > len(shape):
        raise InvalidArgumentError(
            "When broadcasting, the number of dimension of array cannot be greater than in the new shape")

    # Prepend ones to shape and zeros to stride in order to make them the same lengths as `shape`
    ret_shape = [1] * (len(shape) - ary.ndim) + ary.shape
    ret_stride = [0] * (len(shape) - ary.ndim) + ary.stride

    # Broadcast each dimension by setting ret_stride to zero and ret_shape to `shape`
    for i in range(len(ret_shape)):
        if ret_shape[i] != shape[i]:
            if ret_shape[i] == 1:
                ret_shape[i] = shape[i]
                ret_stride[i] = 0
            else:
                raise InvalidArgumentError("Cannot broadcast shape %s to %s" % (ary.shape, shape))
    return bharray.BhArray(ret_shape, ary.dtype, stride=ret_stride, offset=ary.offset, base=ary.base)


def _get_dtyp1e_list(operand_list):
    dtype_list = []
    for op in operand_list:
        if isinstance(op, numbers.Number):
            dtype_list.append(None)
        else:
            dtype_list.append(op.base.dtype)

    input_dtype = None
    for dtype in reversed(dtype_list):
        if dtype is not None:
            input_dtype = dtype
            break

    for i in range(len(dtype_list)):
        if dtype_list[i] is None:
            dtype_list[i] = input_dtype
    return dtype_list


def _call_bh_api_op(op_id, out_operand, in_operand_list):
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
            if op.shape == out_operand.shape:
                handle_list.append(op._bhc_handle)
            else:
                handle_list.append(broadcast_to(op, out_operand.shape)._bhc_handle)
    _bh_api.op(op_id, dtype_enum_list, handle_list)


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
                in_operands[i] = in_operands[i].astype(in_dtype, copy=False)

        # If the output is specified, its shape must match `out_shape`
        if out_operand is None:
            out_operand = bharray.BhArray(out_shape, out_dtype)
        elif out_operand.shape != out_shape:
            raise InvalidArgumentError("The output argument should have the shape: %s" % out_shape)

        if out_dtype == out_operand.dtype:
            _call_bh_api_op(self.info["id"], out_operand, in_operands)
        else:
            tmp_out = bharray.BhArray(out_shape, out_dtype)
            _call_bh_api_op(self.info["id"], tmp_out, in_operands)
            _call_bh_api_op(_info.op["identity"]["id"], out_operand, [tmp_out])
        return out_operand


def generate_ufuncs():
    ufuncs = {}
    for op in _info.op.values():
        if op['elementwise']:
            # Bohrium divide is like division in C/C++ where floats are like
            # `true_divide` and integers are like `floor_divide` in NumPy
            if op['name'] == 'divide':
                op = op.copy()
                op['name'] = 'bh_divide'

            f = Ufunc(op)
            ufuncs[f.info['name']] = f

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
                ufuncs['identity'](ret, out)
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
                ufuncs['identity'](ret, out)
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
