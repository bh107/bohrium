"""
Interop NumPy
~~~~~~~~~~~~~
"""
import numpy_force as numpy
from .bhary import get_bhc
from .target import get_data_pointer


def get_buffer(bh_ary):
    """Return a Buffer object wrapping the memory of the Bohrium array `ary`"""
    mem_ptr = get_data_pointer(get_bhc(bh_ary), copy2host=True)
    return numpy.core.multiarray.int_asbuffer(mem_ptr, bh_ary.nbytes)


def get_array(bh_ary):
    """Return a NumPy array wrapping the memory of the Bohrium array `ary`
       NB: Changing or deallocating `bh_ary` invalidates the return NumPy array
    """
    return numpy.frombuffer(get_buffer(bh_ary), bh_ary.dtype)
