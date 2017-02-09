"""
Masking routines
===========================

"""
import warnings
from . import array_create
from . import bhary
import numpy_force as numpy
from .bhary import fix_returned_biclass

@fix_returned_biclass
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

    if not (bhary.check(condition) or bhary.check(x) or bhary.check(y)):
        return numpy.where(condition, x, y)

    if x is None or y is None:
        warnings.warn("Bohrium only supports where() when 'x' and 'y' are specified")
        return numpy.where(condition, x, y)

    # Let's find a non-scalar and make sure that non-scalars are Bohrium arrays
    t = None
    if not numpy.isscalar(condition):
        condition = array_create.array(condition)
        t = condition
    if not numpy.isscalar(x):
        x = array_create.array(x)
        t = x
    if not numpy.isscalar(y):
        y = array_create.array(y)
        t = y
    if t is None: # All arguments are scalars
        if condition:
            return x
        else:
            return y
    ret = array_create.zeros_like(t)
    ret += condition * x
    ret += ~condition * y
    return ret

