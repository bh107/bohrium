import bh107
import numpy as np

import pytest

UNARY_FUNCS = (
    'argmin',
    'argmax',
    'sort',
    'argsort',
    'mean',
    'copy',
    'ravel',
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
    'int16',
    'int32',
    'int64',
    'uint8',
    'uint16',
    'uint32',
    'uint64'
)

ONLY_FLOAT_SUPPORT = (
    'sin',
    'cos',
    'tan',
    'sinh',
    'cosh',
    'tanh',
    'arcsin',
    'arccos',
    'arctan',
    'arcsinh',
    'arccosh',
    'arctanh'
)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('func', UNARY_FUNCS)
def test_unary_array_func(func, shape, dtype):
    arr = (100 * bh107.random.rand(*shape)).astype(dtype)

    func_handle = getattr(np, func)

    res_bh107 = func_handle(arr)
    res_np = func_handle(arr.copy2numpy())
    assert res_bh107.dtype == res_np.dtype

    if isinstance(res_bh107, bh107.BhArray):
        res_bh107 = res_bh107.copy2numpy()

    try:
        rtol = 10 * np.finfo(res_bh107.dtype).eps
    except ValueError:
        rtol = 0

    np.testing.assert_allclose(
        res_bh107, res_np, rtol=rtol, atol=0
    )


@pytest.mark.parametrize('dtype', ('float32', 'float64'))
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('func', ONLY_FLOAT_SUPPORT)
def test_unary_array_func(func, shape, dtype):
    arr = (100 * bh107.random.rand(*shape)).astype(dtype)

    func_handle = getattr(np, func)

    res_bh107 = func_handle(arr)
    res_np = func_handle(arr.copy2numpy())
    assert res_bh107.dtype == res_np.dtype

    if isinstance(res_bh107, bh107.BhArray):
        res_bh107 = res_bh107.copy2numpy()

    try:
        rtol = 10 * np.finfo(res_bh107.dtype).eps
    except ValueError:
        rtol = 0

    np.testing.assert_allclose(
        res_bh107, res_np, rtol=rtol, atol=0
    )


# explicitly test some special functions

@pytest.mark.parametrize('shape', SHAPES)
def test_unary_where(shape):
    arr = 100 * bh107.random.rand(*shape)

    res_bh107 = np.where(arr > 50)
    res_np = np.where(arr.copy2numpy() > 50)

    assert len(res_bh107) == len(res_np)

    for rb, rn in zip(res_bh107, res_np):
        np.testing.assert_array_equal(
            rb, rn
        )


@pytest.mark.parametrize('shape', SHAPES)
def test_triadic_where(shape):
    arr = 100 * bh107.random.rand(*shape)

    res_bh107 = np.where(arr > 50, 0, 1)
    res_np = np.where(arr.copy2numpy() > 50, 0, 1)

    np.testing.assert_array_equal(
        res_bh107, res_np
    )
