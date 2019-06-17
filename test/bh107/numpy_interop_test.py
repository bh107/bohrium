import contextlib

import pytest

import numpy as np


# conversions through array_interface, asnumpy, copy2numpy

def test_implicit_conversion():
    import bh107

    a = bh107.array([1, 2, 3])
    a_np = np.array([1, 2, 3])

    assert np.array_equal(
        np.array(a),
        a_np
    )


@pytest.mark.parametrize('copy', [True, False])
def test_implicit_conversion_write(copy):
    import bh107

    a = bh107.array([1, 2, 3])
    a_np = np.array(a, copy=copy)
    a_np[...] = 0.

    if copy:
        res = np.array([1, 2, 3])
    else:
        res = np.array([0, 0, 0])

    assert np.array_equal(
        a.copy2numpy(),
        res
    )


def test_asnumpy():
    import bh107

    a = bh107.array([1, 2, 3])
    a_np = np.array([1, 2, 3])

    assert np.array_equal(
        a.asnumpy(),
        a_np
    )


def test_asnumpy_write():
    import bh107

    a = bh107.array([1, 2, 3])
    a.asnumpy()[...] = 0
    a_np = np.array([0, 0, 0])

    assert np.array_equal(
        a.asnumpy(),
        a_np
    )


def test_copy2numpy():
    import bh107

    a = bh107.array([1, 2, 3])
    a_np = np.array([1, 2, 3])

    assert np.array_equal(
        a.copy2numpy(),
        a_np
    )


def test_copy2numpy_write():
    import bh107

    a = bh107.array([1, 2, 3])
    a.copy2numpy()[...] = 0
    a_np = np.array([1, 2, 3])

    assert np.array_equal(
        a.copy2numpy(),
        a_np
    )


# test warrnings and exceptions on fallback

@contextlib.contextmanager
def set_fallback_behavior(behavior):
    import os
    oldval = os.environ.get('BH107_ON_NUMPY_FALLBACK')
    try:
        os.environ['BH107_ON_NUMPY_FALLBACK'] = behavior
        yield
    finally:
        if oldval is None:
            del os.environ['BH107_ON_NUMPY_FALLBACK']
        else:
            os.environ['BH107_ON_NUMPY_FALLBACK'] = oldval


def test_invalid_fallback_value():
    import bh107

    with set_fallback_behavior('foo'), pytest.warns(UserWarning) as record:
        np.array(bh107.array([1]))

    assert len(record) == 2
    message = 'invalid value for environment variable'
    assert message in record[0].message.args[0].lower()


@pytest.mark.parametrize('mode', ['ignore', 'warn', 'raise'])
def test_fallback_in_array_interface(mode):
    import bh107
    from bh107.exceptions import ImplicitConversionWarning

    if mode == 'ignore':
        ctx = pytest.warns(None)
    else:
        # __array_interface__ never raises an exceptions, warns instead
        ctx = pytest.warns(ImplicitConversionWarning)

    with set_fallback_behavior(mode), ctx as record:
        np.array(bh107.array([1]))

    if mode == 'ignore':
        assert not record
    else:
        assert len(record) == 1
        assert 'implicit fallback' in record[0].message.args[0].lower()


@pytest.mark.parametrize('mode', ['ignore', 'warn', 'raise'])
def test_fallback_in_array_ufunc(mode):
    import bh107
    from bh107.exceptions import ImplicitConversionWarning, ImplicitConversionError

    if mode == 'ignore':
        ctx = pytest.warns(None)
    elif mode == 'warn':
        ctx = pytest.warns(ImplicitConversionWarning)
    elif mode == 'raise':
        ctx = pytest.raises(ImplicitConversionError)

    with set_fallback_behavior(mode), ctx as record:
        np.cbrt(bh107.array([1]))

    if mode == 'ignore':
        assert not record
    elif mode == 'warn':
        assert len(record) == 1
        assert 'ufunc "cbrt"' in record[0].message.args[0].lower()
    elif mode == 'raise':
        assert 'ufunc "cbrt"' in str(record.value).lower()


@pytest.mark.parametrize('mode', ['ignore', 'warn', 'raise'])
def test_fallback_in_array_function(mode):
    import bh107
    from bh107.exceptions import ImplicitConversionWarning, ImplicitConversionError

    if mode == 'ignore':
        ctx = pytest.warns(None)
    elif mode == 'warn':
        ctx = pytest.warns(ImplicitConversionWarning)
    elif mode == 'raise':
        ctx = pytest.raises(ImplicitConversionError)

    with set_fallback_behavior(mode), ctx as record:
        np.pad(bh107.array([1]), 1, 'constant')

    if mode == 'ignore':
        assert not record
    elif mode == 'warn':
        assert len(record) == 1
        assert 'function "pad"' in record[0].message.args[0].lower()
    elif mode == 'raise':
        assert 'function "pad"' in str(record.value).lower()


def test_native_function():
    import bh107

    arr = bh107.random.rand(10, 10)

    with set_fallback_behavior('warn'), pytest.warns(None) as record:
        res_bh107 = bh107.mean(arr)
        res_np = np.mean(arr)

    assert not record, record[0].message.args[0]
    assert res_bh107 == res_np
