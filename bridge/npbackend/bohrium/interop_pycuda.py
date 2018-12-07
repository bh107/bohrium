"""
Interop PyCUDA
~~~~~~~~~~~~~~
"""
from bohrium_api import stack_info
from .bhary import get_base
from ._bh import get_data_pointer
from .backend_messaging import cuda_use_current_context
from . import contexts


_iniziated = False


def _import_pycuda_module():
    """Help function to import PyCUDA and checks that a CUDA backend is present"""
    try:
        import pycuda
    except ImportError:
        raise ImportError("Failed to import the `pycuda` module, please install PyCUDA")

    if not stack_info.is_cuda_in_stack():
        raise RuntimeError("No CUDA device in the Bohrium stack! "
                           "Try defining the environment variable `BH_STACK=cuda`.")

    if stack_info.is_proxy_in_stack():
        raise RuntimeError("Cannot directly access the CUDA device through a proxy.")
    return pycuda


def available():
    """Is CUDA available?"""
    try:
        _import_pycuda_module()
        return True
    except ImportError:
        return False
    except RuntimeError:
        return False


def init():
    """Initiate the PyCUDA module. Must be called before any other PyCUDA calls and
    preferable also before any Bohrium operations"""
    global _iniziated
    if _iniziated:
        return
    _initiated = True
    if not available():
        return
    from . import _bh
    _bh.flush()
    import pycuda
    import pycuda.autoinit
    pycuda.driver.mem_alloc(1)  # Force PyCUDA to activate the context as "current context"
    cuda_use_current_context()  # And then tell Bohrium to use that context


def get_gpuarray(bh_ary):
    """Return a PyCUDA GPUArray object that points to the same device memory as `bh_ary`.

    Parameters
    ----------
    bh_ary : ndarray (Bohrium array)
        Must be a Bohrium base array

    Returns
    -------
    out : GPUArray

    Notes
    -----
    Changing or deallocating `bh_ary` invalidates the returned GPUArray array!

    """

    if get_base(bh_ary) is not bh_ary:
        raise RuntimeError('`bh_ary` must be a base array and not a view')
    assert (bh_ary.bhc_mmap_allocated)

    with contexts.DisableBohrium():
        _import_pycuda_module()
        from pycuda import gpuarray
        dev_ptr = get_data_pointer(get_base(bh_ary), copy2host=False, allocate=True)
        return gpuarray.GPUArray(bh_ary.shape, bh_ary.dtype, gpudata=dev_ptr)


def max_local_memory(cuda_device=None):
    """Returns the maximum allowed local memory (memory per block) on `cuda_device`.
       If `cuda_device` is None, use current device"""

    pycuda = _import_pycuda_module()
    if cuda_device is None:
        cuda_device = pycuda.driver.Context.get_device()
    return cuda_device.get_attributes()[pycuda.driver.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]


def type_np2cuda_str(np_type):
    """Converts a NumPy type to a CUDA type string"""
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
