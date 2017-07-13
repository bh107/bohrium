"""
Interop PyOpenCL
~~~~~~~~~~~~~~~~
"""
from .bhary import get_bhc
from .target import get_data_pointer, get_device_context

_opencl_context = None


def get_context():
    """Return a PyOpenCL context"""
    import pyopencl
    cxt = get_device_context()
    if cxt is None:
        raise RuntimeError("Do OpenCL device in the Bohrium stack! "
                           "Try defining the environment variable `BH_STACK=opencl`.")
    return pyopencl.Context.from_int_ptr(cxt)


def import_pyopencl_module():
    """Help function to import PyOpenCL and checks that a OpenCL backend is present"""
    try:
        import pyopencl
    except ImportError:
        raise ImportError("Failed to import the `pyopencl` module, please install PyOpenCL")

    global _opencl_context
    if _opencl_context is None:
        _opencl_context = get_device_context()  # Fails when no OpenCL backend is present
    return pyopencl


def get_buffer(ary):
    """Return a OpenCL Buffer object wrapping the Bohrium array `ary`"""
    pyopencl = import_pyopencl_module()
    cl_mem = get_data_pointer(get_bhc(ary), copy2host=False)
    return pyopencl.Buffer.from_cl_mem_as_int(cl_mem)

