"""
Interop NumPy
~~~~~~~~~~~~~
"""
import numpy_force as numpy
from .bhary import get_bhc
from .target import get_data_pointer


def get_array(bh_ary):
    """Return a NumPy array wrapping the memory of the Bohrium array `ary`
       NB: Changing or deallocating `bh_ary` invalidates the returned NumPy array
    """
    return bh_ary._numpy_wrapper()
