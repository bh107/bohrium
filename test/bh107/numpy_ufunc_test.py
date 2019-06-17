import bh107
import numpy as np

import pytest

UNARY_UFUNCS = (
    'min',
    'max',
    'abs',
    'sum',
    'cumsum',
    'add.reduce',
    'add.accumulate',
    'multiply.reduce',
    'multiply.accumulate',
    'prod',
    'cumprod',
    'logical_not',
)

BINARY_UFUNCS = (
    'add',
    'subtract',
    'divide',
    'multiply',
    'logical_and',
    'logical_or',
    'logical_xor',
    'equal',
    'not_equal',
    'less',
    'less_equal',
    'greater',
    'greater_equal',
)

SHAPES = (
    (1,),
    (10,),
    (10, 10),
    (1,) * 16,
)

DTYPES = (
    'bool',
    'float32',
    'float64',
    'int8',
    'int16'
)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('ufunc', UNARY_UFUNCS)
def test_unary_array_ufunc(ufunc, shape, dtype):
    if 'logical' in ufunc and dtype != 'bool':
        return

    arr = (100 * bh107.random.rand(*shape)).astype(dtype)
    if '.' in ufunc:
        ufunc, method = ufunc.split('.')
    else:
        method = '__call__'

    func_handle = getattr(getattr(np, ufunc), method)

    res_bh107 = func_handle(arr)
    if isinstance(res_bh107, bh107.BhArray):
        res_bh107 = res_bh107.copy2numpy()

    res_np = func_handle(arr.asnumpy()).astype(res_bh107.dtype)

    try:
        rtol = 10 * np.finfo(res_bh107.dtype).eps
    except ValueError:
        rtol = 0

    np.testing.assert_allclose(
        res_bh107, res_np, rtol=rtol, atol=0
    )


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('ufunc', BINARY_UFUNCS)
def test_binary_array_ufunc(ufunc, shape, dtype):
    if 'logical' in ufunc and dtype != 'bool':
        return

    if ufunc in ('subtract',) and dtype == 'bool':
        return

    arr1 = (100 * bh107.random.rand(*shape)).astype(dtype)
    arr2 = arr1.copy()

    if '.' in ufunc:
        ufunc, method = ufunc.split('.')
    else:
        method = '__call__'

    func_handle = getattr(getattr(np, ufunc), method)

    res_bh107 = func_handle(arr1, arr2)
    if isinstance(res_bh107, bh107.BhArray):
        res_bh107 = res_bh107.copy2numpy()

    res_np = func_handle(arr1.asnumpy(), arr2.asnumpy()).astype(res_bh107.dtype)

    try:
        rtol = 10 * np.finfo(res_bh107.dtype).eps
    except ValueError:
        rtol = 0

    np.testing.assert_allclose(
        res_bh107, res_np, rtol=rtol, atol=0
    )
