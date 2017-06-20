"""
Masking routines
===========================

"""
import warnings
from . import array_create
from . import bhary
from . import reorganization
from . import array_manipulation
from . import ufuncs
from . import summations
import numpy_force as numpy
from .bhary import fix_biclass_wrapper


@fix_biclass_wrapper
def where(condition, x=None, y=None):
    """
    where(condition, [x, y])

    Return elements, either from `x` or `y`, depending on `condition`.

    If only `condition` is given, return ``condition.nonzero()``.

    Parameters
    ----------
    condition : array_like, bool
        When True, yield `x`, otherwise yield `y`.
    x, y : array_like, optional
        Values from which to choose. `x` and `y` need to have the same
        shape as `condition`.

    Returns
    -------
    out : ndarray or tuple of ndarrays
        If both `x` and `y` are specified, the output array contains
        elements of `x` where `condition` is True, and elements from
        `y` elsewhere.

        If only `condition` is given, return the tuple
        ``condition.nonzero()``, the indices where `condition` is True.

    See Also
    --------
    nonzero, choose

    Notes
    -----
    If `x` and `y` are given and input arrays are 1-D, `where` is
    equivalent to::

        [xv if c else yv for (c,xv,yv) in zip(condition,x,y)]

    Examples
    --------
    >>> np.where([[True, False], [True, True]],
    ...          [[1, 2], [3, 4]],
    ...          [[9, 8], [7, 6]])
    array([[1, 8],
           [3, 4]])

    >>> np.where([[0, 1], [1, 0]])
    (array([0, 1]), array([1, 0]))

    >>> x = np.arange(9.).reshape(3, 3)
    >>> np.where( x > 5 )
    (array([2, 2, 2]), array([0, 1, 2]))
    >>> x[np.where( x > 3.0 )]               # Note: result is 1D.
    array([ 4.,  5.,  6.,  7.,  8.])
    >>> np.where(x < 5, x, -1)               # Note: broadcasting.
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -1.],
           [-1., -1., -1.]])

    Find the indices of elements of `x` that are in `goodvalues`.

    >>> goodvalues = [3, 4, 7]
    >>> ix = np.in1d(x.ravel(), goodvalues).reshape(x.shape)
    >>> ix
    array([[False, False, False],
           [ True,  True, False],
           [False,  True, False]], dtype=bool)
    >>> np.where(ix)
    (array([1, 1, 2]), array([0, 1, 1]))

    """
    if x is None or y is None:
        warnings.warn("Bohrium only supports where() when 'x' and 'y' are specified", stacklevel=2)
        return numpy.where(condition)

    if not (bhary.check(condition) or bhary.check(x) or bhary.check(y)):
        return numpy.where(condition, x, y)

    # Make sure that non-scalars are Bohrium arrays
    if numpy.isscalar(condition):
        condition = bool(condition)
    else:
        condition = array_create.array(condition).astype("bool")

    if not numpy.isscalar(x):
        x = array_create.array(x)

    if not numpy.isscalar(y):
        y = array_create.array(y)

    # Shortcut if all arguments are scalars
    if all(numpy.isscalar(k) or k.size == 1 for k in (x, y, condition)):
        return x if condition else y

    # Find appropriate output type
    array_types = []
    scalar_types = []
    for v in (x, y):
        if numpy.isscalar(v):
            scalar_types.append(type(v))
        else:
            array_types.append(v.dtype)
    out_type = numpy.find_common_type(array_types, scalar_types)

    # Shortcut if input arrays are finite
    if ufuncs.isfinite(x).all() and ufuncs.isfinite(y).all():
        if numpy.isscalar(condition):
            res = condition * x + (not condition) * y
        else:
            res = condition * x + ufuncs.logical_not(condition) * y
        if numpy.isscalar(res):
            return out_type(res)
        else:
            return res.astype(out_type)

    # General case: use fancy indexing
    (condition, x, y), newshape = array_manipulation.broadcast_arrays(condition, x, y)
    ret = array_create.zeros(newshape, dtype=out_type)
    ret[condition] = x if numpy.isscalar(x) else x[condition]
    ret[~condition] = y if numpy.isscalar(y) else y[~condition]
    return ret


def masked_get(ary, bool_mask):
    """
    Get the elements of 'ary' specified by 'bool_mask'.
    """

    return ary[reorganization.nonzero(bool_mask)]


def masked_set(ary, bool_mask, value):
    """
    Set the 'value' into 'ary' at the location specified through 'bool_mask'.
    """

    if numpy.isscalar(value) and ufuncs.isfinite(value):
        ary *= ~bool_mask
        ary += bool_mask * value
    else:
        ary[reorganization.nonzero(bool_mask)] = value
