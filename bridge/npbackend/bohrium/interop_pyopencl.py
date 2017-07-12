"""
Interop PyOpenCL
~~~~~~~~~~~~~~~~
"""
from .bhary import get_bhc
from .target import get_data_pointer, get_device_context


def get_context():
    """Return a PyOpenCL context"""
    import pyopencl
    cxt = get_device_context()
    if cxt is None:
        raise RuntimeError("Do OpenCL device")
    return pyopencl.Context.from_int_ptr(cxt)


def get_buffer(ary):
    """Return a OpenCL Buffer object wrapping the Bohrium array `ary`"""
    import pyopencl
    cl_mem = get_data_pointer(get_bhc(ary), copy2host=False)
    return pyopencl.Buffer.from_cl_mem_as_int(cl_mem)

