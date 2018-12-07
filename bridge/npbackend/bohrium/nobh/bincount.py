import math
import numpy_force as np
from bohrium_api import stack_info
from .. import interop_pyopencl
from .. import interop_pycuda
from .. import array_create
from .bincount_cython import bincount_cython


def bincount(x, weights=None, minlength=None):
    """Count number of occurrences of each value in array of non-negative ints.

    The number of bins (of size 1) is one larger than the largest value in
    `x`. If `minlength` is specified, there will be at least this number
    of bins in the output array (though it will be longer if necessary,
    depending on the contents of `x`).
    Each bin gives the number of occurrences of its index value in `x`.
    If `weights` is specified the input array is weighted by it, i.e. if a
    value ``n`` is found at position ``i``, ``out[n] += weight[i]`` instead
    of ``out[n] += 1``.

    Parameters
    ----------
    x : array_like, 1 dimension, nonnegative ints
        Input array.
    weights : array_like, optional
        Weights, array of the same shape as `x`.
    minlength : int, optional
        A minimum number of bins for the output array.

        .. versionadded:: 1.6.0

    Returns
    -------
    out : ndarray of ints
        The result of binning the input array.
        The length of `out` is equal to ``np.amax(x)+1``.

    Raises
    ------
    ValueError
        If the input is not 1-dimensional, or contains elements with negative
        values, or if `minlength` is non-positive.
    TypeError
        If the type of the input is float or complex.

    See Also
    --------
    histogram, digitize, unique

    Examples
    --------
    >>> np.bincount(np.arange(5))
    array([1, 1, 1, 1, 1])
    >>> np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
    array([1, 3, 1, 1, 0, 0, 0, 1])

    >>> x = np.array([0, 1, 1, 3, 2, 1, 7, 23])
    >>> np.bincount(x).size == np.amax(x)+1
    True

    The input array needs to be of integer dtype, otherwise a
    TypeError is raised:

    >>> np.bincount(np.arange(5, dtype=np.float))
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: array cannot be safely cast to required type

    A possible use of ``bincount`` is to perform sums over
    variable-size chunks of an array, using the ``weights`` keyword.

    >>> w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6]) # weights
    >>> x = np.array([0, 1, 1, 2, 2, 2])
    >>> np.bincount(x,  weights=w)
    array([ 0.3,  0.7,  1.1])

    """

    # Let's find the backend to handle the bincount
    x = array_create.array(x)
    assert(np.issubdtype(x.dtype.type, np.integer))
    assert(np.issubdtype(x.dtype.type, np.integer))

    if stack_info.is_proxy_in_stack():  # Cannot directly access array data through a proxy
        return np.bincount(x.copy2numpy(), weights=weights, minlength=minlength)

    try:
        if weights is not None:
            raise NotImplementedError("OpenCL doesn't support the `weights` argument")
        return bincount_pyopencl(x, minlength=minlength)
    except NotImplementedError:
        try:
            if weights is not None:
                raise NotImplementedError("CUDA doesn't support the `weights` argument")
            return bincount_pycuda(x, minlength=minlength)
        except NotImplementedError:
            try:
                return bincount_cython(x, weights=weights, minlength=minlength)
            except NotImplementedError:
                return np.bincount(x.copy2numpy(), weights=weights, minlength=minlength)


def bincount_pyopencl(x, minlength=None):
    """PyOpenCL implementation of `bincount()`"""

    if not interop_pyopencl.available():
        raise NotImplementedError("OpenCL not available")

    import pyopencl as cl
    ctx = interop_pyopencl.get_context()
    queue = cl.CommandQueue(ctx)

    x_max = int(x.max())
    if x_max < 0:
        raise RuntimeError("bincount(): first argument must be a 1 dimensional, non-negative int array")
    if x_max > np.iinfo(np.uint32).max:
        raise NotImplementedError("OpenCL: the elements in the first argument must fit in a 32bit integer")
    if minlength is not None:
        x_max = max(x_max, minlength)

    # TODO: handle large max element by running multiple bincount() on a range
    if x_max >= interop_pyopencl.max_local_memory(queue.device) // x.itemsize:
        raise NotImplementedError("OpenCL: max element is too large for the GPU")

    # Let's create the output array and retrieve the in-/output OpenCL buffers
    # NB: we always return uint32 array
    ret = array_create.empty((x_max+1, ), dtype=np.uint32)
    x_buf = interop_pyopencl.get_buffer(x)
    ret_buf = interop_pyopencl.get_buffer(ret)

    # OpenCL kernel is based on the book "OpenCL Programming Guide" by Aaftab Munshi at al.
    source = """
    kernel void histogram_partial(
        global DTYPE *input,
        global uint *partial_histo,
        uint input_size
    ){
        int local_size = (int)get_local_size(0);
        int group_indx = get_group_id(0) * HISTO_SIZE;
        int gid = get_global_id(0);
        int tid = get_local_id(0);

        local uint tmp_histogram[HISTO_SIZE];

        int j = HISTO_SIZE;
        int indx = 0;

        // clear the local buffer that will generate the partial histogram
        do {
            if (tid < j)
                tmp_histogram[indx+tid] = 0;
            j -= local_size;
            indx += local_size;
        } while (j > 0);

        barrier(CLK_LOCAL_MEM_FENCE);

        if (gid < input_size) {
            atomic_inc(&tmp_histogram[input[gid]]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // copy the partial histogram to appropriate location in
        // histogram given by group_indx
        if (local_size >= HISTO_SIZE){
            if (tid < HISTO_SIZE)
                partial_histo[group_indx + tid] = tmp_histogram[tid];
        }else{
            j = HISTO_SIZE;
            indx = 0;
            do {
                if (tid < j)
                    partial_histo[group_indx + indx + tid] = tmp_histogram[indx + tid];

                j -= local_size;
                indx += local_size;
            } while (j > 0);
        }
    }

    kernel void histogram_sum_partial_results(
        global uint *partial_histogram,
        int num_groups,
        global uint *histogram
    ){
        int gid = (int)get_global_id(0);
        int group_indx;
        int n = num_groups;
        local uint tmp_histogram[HISTO_SIZE];

        tmp_histogram[gid] = partial_histogram[gid];
        group_indx = HISTO_SIZE;
        while (--n > 0) {
            tmp_histogram[gid] += partial_histogram[group_indx + gid];
            group_indx += HISTO_SIZE;
        }
        histogram[gid] = tmp_histogram[gid];
    }
    """
    source = source.replace("HISTO_SIZE", "%d" % ret.shape[0])
    source = source.replace("DTYPE", interop_pyopencl.type_np2opencl_str(x.dtype))
    prg = cl.Program(ctx, source).build()

    # Calculate sizes for the kernel execution
    local_size = interop_pyopencl.kernel_info(prg.histogram_partial, queue)[0]  # Max work-group size
    num_groups = int(math.ceil(x.shape[0] / float(local_size)))
    global_size = local_size * num_groups

    # First we compute the partial histograms
    partial_res_g = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, num_groups * ret.nbytes)
    prg.histogram_partial(queue, (global_size,), (local_size,), x_buf, partial_res_g, np.uint32(x.shape[0]))

    # Then we sum the partial histograms into the final histogram
    prg.histogram_sum_partial_results(queue, ret.shape, None, partial_res_g, np.uint32(num_groups), ret_buf)
    return ret


def bincount_pycuda(x, minlength=None):
    """PyCUDA implementation of `bincount()`"""

    if not interop_pycuda.available():
        raise NotImplementedError("CUDA not available")

    import pycuda
    from pycuda.compiler import SourceModule

    interop_pycuda.init()

    x_max = int(x.max())
    if x_max < 0:
        raise RuntimeError("bincount(): first argument must be a 1 dimensional, non-negative int array")
    if x_max > np.iinfo(np.uint32).max:
        raise NotImplementedError("CUDA: the elements in the first argument must fit in a 32bit integer")
    if minlength is not None:
        x_max = max(x_max, minlength)

    # TODO: handle large max element by running multiple bincount() on a range
    if x_max >= interop_pycuda.max_local_memory() // x.itemsize:
        raise NotImplementedError("CUDA: max element is too large for the GPU")

    # Let's create the output array and retrieve the in-/output OpenCL buffers
    # NB: we always return uint32 array
    ret = array_create.ones((x_max+1, ), dtype=np.uint32)
    x_buf = interop_pycuda.get_gpuarray(x)
    ret_buf = interop_pycuda.get_gpuarray(ret)

    # CUDA kernel is based on the book "OpenCL Programming Guide" by Aaftab Munshi at al.
    source = """
    __global__ void histogram_partial(
        DTYPE *input,
        uint *partial_histo,
        uint input_size
    ){
        int local_size = blockDim.x;
        int group_indx = blockIdx.x * HISTO_SIZE;
        int gid = (blockIdx.x * blockDim.x + threadIdx.x);
        int tid = threadIdx.x;

        __shared__ uint tmp_histogram[HISTO_SIZE];

        int j = HISTO_SIZE;
        int indx = 0;

        // clear the local buffer that will generate the partial histogram
        do {
            if (tid < j)
                tmp_histogram[indx+tid] = 0;
            j -= local_size;
            indx += local_size;
        } while (j > 0);

        __syncthreads();

        if (gid < input_size) {
            atomicAdd(&tmp_histogram[input[gid]], 1);
        }

        __syncthreads();

        // copy the partial histogram to appropriate location in
        // histogram given by group_indx
        if (local_size >= HISTO_SIZE){
            if (tid < HISTO_SIZE)
                partial_histo[group_indx + tid] = tmp_histogram[tid];
        }else{
            j = HISTO_SIZE;
            indx = 0;
            do {
                if (tid < j)
                    partial_histo[group_indx + indx + tid] = tmp_histogram[indx + tid];

                j -= local_size;
                indx += local_size;
            } while (j > 0);
        }
    }

    __global__ void histogram_sum_partial_results(
        uint *partial_histogram,
        int num_groups,
        uint *histogram
    ){
        int gid = (blockIdx.x * blockDim.x + threadIdx.x);
        int group_indx;
        int n = num_groups;
        __shared__ uint tmp_histogram[HISTO_SIZE];

        tmp_histogram[gid] = partial_histogram[gid];
        group_indx = HISTO_SIZE;
        while (--n > 0) {
            tmp_histogram[gid] += partial_histogram[group_indx + gid];
            group_indx += HISTO_SIZE;
        }
        histogram[gid] = tmp_histogram[gid];
    }
    """
    source = source.replace("HISTO_SIZE", "%d" % ret.shape[0])
    source = source.replace("DTYPE", interop_pycuda.type_np2cuda_str(x.dtype))
    prg = SourceModule(source)

    # Calculate sizes for the kernel execution
    kernel = prg.get_function("histogram_partial")
    local_size = kernel.get_attribute(pycuda.driver.function_attribute.MAX_THREADS_PER_BLOCK)  # Max work-group size
    num_groups = int(math.ceil(x.shape[0] / float(local_size)))
    global_size = local_size * num_groups

    # First we compute the partial histograms
    partial_res_g = pycuda.driver.mem_alloc(num_groups * ret.nbytes)
    kernel(x_buf, partial_res_g, np.uint32(x.shape[0]), block=(local_size, 1, 1), grid=(num_groups, 1))

    # Then we sum the partial histograms into the final histogram
    kernel = prg.get_function("histogram_sum_partial_results")
    kernel(partial_res_g, np.uint32(num_groups), ret_buf, block=(1, 1, 1), grid=(ret.shape[0], 1))
    return ret  


