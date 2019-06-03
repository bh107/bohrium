import bh107
import numpy as np

import pytest

UNARY_FUNCS = (
    'argmin',
    'argmax',
    'sort',
    'argsort',
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
    'arctanh',
    'mean',
    'copy',
    'ravel'
)

SHAPES = (
    (1,),
    (10,),
    (10, 10),
    #(1,) * 100,
)

DTYPES = (
    'bool',
    #'float16',
    'float32',
    'float64',
    'int8',
    'int16'
)

NO_BOOL_SUPPORT = (
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
    if func in NO_BOOL_SUPPORT and dtype == 'bool':
        return

    arr = (100 * bh107.random.rand(*shape)).astype(dtype)

    func_handle = getattr(np, func)

    res_bh107 = func_handle(arr).astype(dtype)
    if isinstance(res_bh107, bh107.BhArray):
        res_bh107 = res_bh107.copy2numpy()

    res_np = func_handle(arr.asnumpy()).astype(dtype)

    try:
        rtol = 10 * np.finfo(arr.dtype).eps
    except ValueError:
        rtol = 0

    np.testing.assert_allclose(
        res_bh107, res_np, rtol=rtol, atol=0
    )
