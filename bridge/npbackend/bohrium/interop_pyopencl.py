"""
Interop PyOpenCL
~~~~~~~~~~~~~~~~
"""
from .bhary import get_bhc
from .target import get_data_pointer, get_device_context, set_data_pointer
from .backend_messaging import runtime_info

_opencl_is_in_stack = None


def _is_opencl_in_stack():
    global _opencl_is_in_stack
    if _opencl_is_in_stack is None:
        _opencl_is_in_stack = "OpenCL" in runtime_info()
    return _opencl_is_in_stack


def _import_pyopencl_module():
    """Help function to import PyOpenCL and checks that a OpenCL backend is present"""
    try:
        import pyopencl
    except ImportError:
        raise ImportError("Failed to import the `pyopencl` module, please install PyOpenCL")

    if not _is_opencl_in_stack():
        raise RuntimeError("No OpenCL device in the Bohrium stack! "
                           "Try defining the environment variable `BH_STACK=opencl`.")
    return pyopencl


def get_context():
    """Return a PyOpenCL context"""
    pyopencl = _import_pyopencl_module()
    cxt = get_device_context()
    if cxt is None:
        raise RuntimeError("No OpenCL device in the Bohrium stack! "
                           "Try defining the environment variable `BH_STACK=opencl`.")
    return pyopencl.Context.from_int_ptr(cxt)


def get_buffer(bh_ary):
    """Return a OpenCL Buffer object wrapping the Bohrium array `ary`"""
    pyopencl = _import_pyopencl_module()
    cl_mem = get_data_pointer(get_bhc(bh_ary), copy2host=False)
    return pyopencl.Buffer.from_int_ptr(cl_mem)


def set_buffer(ary, buffer):
    """Assign a OpenCL Buffer object to a Bohrium array `ary`"""
    set_data_pointer(get_bhc(ary), buffer.int_ptr, host_ptr=False)


def get_array(bh_ary, queue):
    _import_pyopencl_module()
    from pyopencl import array as clarray
    return clarray.Array(queue, bh_ary.shape, bh_ary.dtype, data=get_buffer(bh_ary))
