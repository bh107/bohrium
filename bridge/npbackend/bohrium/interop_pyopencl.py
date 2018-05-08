"""
Interop PyOpenCL
~~~~~~~~~~~~~~~~
"""
from .bhary import get_base
from ._bh import get_data_pointer, set_data_pointer, get_device_context
from . import bh_info


def _import_pyopencl_module():
    """Help function to import PyOpenCL and checks that a OpenCL backend is present"""
    try:
        import pyopencl
    except ImportError:
        raise ImportError("Failed to import the `pyopencl` module, please install PyOpenCL")

    if not bh_info.is_opencl_in_stack():
        raise RuntimeError("No OpenCL device in the Bohrium stack! "
                           "Try defining the environment variable `BH_STACK=opencl`.")

    if bh_info.is_proxy_in_stack():
        raise RuntimeError("Cannot directly access the OpenCL device through a proxy.")
    return pyopencl


def available():
    """Is PyOpenCL available?"""
    try:
        _import_pyopencl_module()
        return True
    except ImportError:
        return False
    except RuntimeError:
        return False


def get_context():
    """Return a PyOpenCL context"""
    pyopencl = _import_pyopencl_module()
    cxt = get_device_context()
    if cxt is None:
        raise RuntimeError("No OpenCL device in the Bohrium stack! "
                           "Try defining the environment variable `BH_STACK=opencl`.")
    return pyopencl.Context.from_int_ptr(cxt)


def get_buffer(bh_ary):
    """Return a OpenCL Buffer object wrapping the Bohrium array `ary`.

    Parameters
    ----------
    bh_ary : ndarray (Bohrium array)
        Must be a Bohrium base array

    Returns
    -------
    out : pyopencl.Buffer

    Notes
    -----
    Changing or deallocating `bh_ary` invalidates the returned pyopencl.Buffer!

    """

    if get_base(bh_ary) is not bh_ary:
        raise RuntimeError('`bh_ary` must be a base array and not a view')
    assert (bh_ary.bhc_mmap_allocated)

    pyopencl = _import_pyopencl_module()
    cl_mem = get_data_pointer(get_base(bh_ary), copy2host=False, allocate=True)
    return pyopencl.Buffer.from_int_ptr(cl_mem)


def set_buffer(bh_ary, buffer):
    """Assign a OpenCL Buffer object to a Bohrium array `ary`.

    Parameters
    ----------
    bh_ary : ndarray (Bohrium array)
        Must be a Bohrium base array

    buffer : pyopencl.Buffer
        The PyOpenCL device buffer
    """

    if get_base(bh_ary) is not bh_ary:
        raise RuntimeError('`bh_ary` must be a base array and not a view')

    set_data_pointer(get_base(bh_ary), buffer.int_ptr, host_ptr=False)


def get_array(bh_ary, queue):
    _import_pyopencl_module()
    from pyopencl import array as clarray
    return clarray.Array(queue, bh_ary.shape, bh_ary.dtype, data=get_buffer(bh_ary))


def kernel_info(opencl_kernel, queue):
    """Info about the `opencl_kernel`
    Returns 4-tuple:
            - Max work-group size
            - Recommended work-group multiple
            - Local mem used by kernel
            - Private mem used by kernel
    """
    cl = _import_pyopencl_module()
    info = cl.kernel_work_group_info
    # Max work-group size
    wg_size = opencl_kernel.get_work_group_info(info.WORK_GROUP_SIZE, queue.device)
    # Recommended work-group multiple
    wg_multiple = opencl_kernel.get_work_group_info(info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, queue.device)
    # Local mem used by kernel
    local_usage = opencl_kernel.get_work_group_info(info.LOCAL_MEM_SIZE, queue.device)
    # Private mem used by kernel
    private_usage = opencl_kernel.get_work_group_info(info.PRIVATE_MEM_SIZE, queue.device)
    return (wg_size, wg_multiple, local_usage, private_usage)


def max_local_memory(opencl_device):
    """Returns the maximum allowed local memory on `opencl_device`"""
    cl = _import_pyopencl_module()
    return opencl_device.get_info(cl.device_info.LOCAL_MEM_SIZE)


def type_np2opencl_str(np_type):
    """Converts a NumPy type to a OpenCL type string"""
    import numpy as np
    if np_type == np.bool:
        return "bool"
    elif np_type == np.int8:
        return "char"
    elif np_type == np.int16:
        return "short"
    if np_type == np.int32:
        return "int"
    elif np_type == np.int64:
        return "long"
    elif np_type == np.uint8:
        return "uchar"
    elif np_type == np.uint16:
        return "ushort"
    elif np_type == np.uint32:
        return "uint"
    elif np_type == np.uint64:
        return "ulong"
    elif np_type == np.float32:
        return "float"
    elif np_type == np.float64:
        return "double"
    else:
        return "UNKNOWN"
