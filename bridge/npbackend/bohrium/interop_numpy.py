"""
Interop NumPy
~~~~~~~~~~~~~
"""
from .bhary import get_base
from . import bh_info


def get_array(bh_ary):
    """Return a NumPy array wrapping the memory of the Bohrium array `ary`.
    
    Parameters
    ----------
    bh_ary : ndarray (Bohrium array)
        Must be a Bohrium base array
    
    Returns
    -------
    out : ndarray (regular NumPy array)

    Notes
    -----
    Changing or deallocating `bh_ary` invalidates the returned NumPy array!

    """

    if bh_info.is_proxy_in_stack():
        raise RuntimeError("Cannot directly access array data through a proxy.")

    if get_base(bh_ary) is not bh_ary:
        raise RuntimeError('`bh_ary` must be a base array and not a view')
    assert(bh_ary.bhc_mmap_allocated)

    return bh_ary._numpy_wrapper()
