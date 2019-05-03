import numpy as np
from bohrium_api import stack_info, _bh_api
from . import bharray, _dtype_util

_default_compiler_command = None


def get_default_compiler_command():
    """ Returns the default compiler command, which is the one typically extended with extra link commands """
    global _default_compiler_command
    if _default_compiler_command is None:
        import re
        from bohrium_api import stack_info
        m = re.search("JIT Command: \"([^\"]*)\"", stack_info.info()['runtime_info'])
        if m is None:
            raise RuntimeError("'JIT Command' not found in the Bohrium backend")
        _default_compiler_command = m.group(1)
    return _default_compiler_command


def execute(kernel_source, operand_list, compiler_command=None, tag="openmp", param=None, only_behaving_operands=True):
    """ Compile and execute the function `execute()` with the arguments in `operand_list`

    Parameters
    ----------
    kernel_source : str
        The kernel source code that most define the function `execute()` that should take arguments corresponding
        to the `operand_list`
    operand_list : list of bohrium arrays
        The arrays given to the `execute()` function defined in `kernel_source`
    compiler_command : str, optional
        The compiler command to use when comping the kernel. `{OUT}` and `{IN}` in the command are replaced with the
        name of the binary and source path.
        When this options isn't specified, the default command are used see `get_default_compiler_command()`.
    tag : str, optional
        Name of the backend that should handle this kernel.
    param : dict, optional
        Backend specific parameters (e.g. OpenCL needs `global_work_size` and `local_work_size`).
    only_behaving_operands : bool, optional
        Set to False in order to allow non-behaving operands. Requirements for a behaving array:
             * Is a bohrium array
             * Is C-style contiguous
             * Points to the first element in the underlying base array (no offset)
             * Has the same total length as its base
        See `make_behaving()`

    Examples
    --------
    # Simple addition kernel
    import bohrium as bh
    kernel = r'''
    #include <stdint.h>
    void execute(double *a, double *b, double *c) {
        for(uint64_t i=0; i<100; ++i) {
            c[i] = a[i] + b[i] + i;
        }
    }'''
    a = bh107.ones(100, np.double)
    b = bh107.ones(100, np.double)
    res = bh107.empty_like(a)
    bh107.user_kernel.execute(kernel, [a, b, res])

    """
    if stack_info.is_proxy_in_stack():
        raise RuntimeError("The proxy backend does not support user kernels")
    if compiler_command is None:
        compiler_command = get_default_compiler_command()

    op_handle_list = []
    for op in operand_list:
        if not isinstance(op, bharray.BhArray):
            raise TypeError("All operands in `operand_list` must be BhArrays")
        if only_behaving_operands and not op.isbehaving():
            raise TypeError("Operand is not behaving set `only_behaving_operands=False` or use `make_behaving()`")
        op_handle_list.append(op._bhc_handle)

    if param is None:
        param = {}

    def parse_param():
        import collections
        for key, value in param.items():
            if isinstance(value, collections.Iterable) and not isinstance(value, str):
                value = " ".join(str(subvalue) for subvalue in value)
            yield "%s: %s" % (key, value)

    param_str = "; ".join(parse_param())

    _bh_api.flush()
    ret_msg = _bh_api.user_kernel(kernel_source, op_handle_list, compiler_command, tag, param_str)
    if len(ret_msg) > 0:
        raise RuntimeError(ret_msg)


def dtype_to_c99(dtype):
    """ Returns the C99 name of `dtype` """
    dtype = _dtype_util.obj_to_dtype(dtype)
    if isinstance(dtype, _dtype_util.integers):
        return "%s_t" % str(dtype)
    elif dtype == np.float32:
        return "float"
    elif dtype == np.float64:
        return "double"
    elif dtype == np.complex64:
        return "float complex"
    elif dtype == np.complex128:
        return "double complex"
    raise TypeError("dtype '%s' unsupported" % str(dtype))


def gen_function_prototype(operand_list, operand_name_list=None):
    """ Returns the `execute() definition based on the arrays in `operand_list` """
    dtype_list = [dtype_to_c99(t.dtype) for t in operand_list]
    ret = "#include <stdint.h>\n#include <complex.h>\n"
    ret += "void execute("
    for i in range(len(dtype_list)):
        ret += "%s *" % dtype_list[i]
        if operand_name_list is None:
            ret += "a%d, " % i
        else:
            ret += "%s, " % operand_name_list[i]
    return "%s)\n" % ret[:-2]


def make_behaving(ary, dtype=None):
    """ Make sure that `ary` is a "behaving" bohrium array of type `dtype`.

    Requirements for a behaving array:
     * Is a bohrium array
     * Points to the first element in the underlying base array (no offset)
     * Has the same total length as its base

    Parameters
    ----------
    ary : BhArray
        The array to make behaving
    dtype : boolean, optional
        The return array is converted to `dtype` if not None
    
    Returns
    -------
    A behaving BhArray that might be a copy of `ary`

    Note
    ----
    Use this function to make sure that operands given to `execute()` is "behaving" that is
    the kernel can access the arrays without worrying about offset and stride.
    """

    if ary.isbehaving():
        ret = ary
    else:
        ret = ary.flatten(always_copy=True)
    if dtype is not None:
        ret = ret.astype(dtype)
    return ret
