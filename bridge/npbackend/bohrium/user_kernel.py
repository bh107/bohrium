import numpy_force as np
from . import _bh, bhary, _util

_default_compiler_command = None


def get_default_compiler_command():
    global _default_compiler_command
    if _default_compiler_command is None:
        import re
        from bohrium_api import stack_info
        m = re.search("JIT Command: \"([^\"]*)\"", stack_info.info()['runtime_info'])
        if m is None:
            raise RuntimeError("'JIT Command' not found in the Bohrium backend")
        _default_compiler_command = m.group(1)
    return _default_compiler_command


def execute(kernel_source, operand_list, compiler_command=None, tag="openmp"):
    if compiler_command is None:
        compiler_command = get_default_compiler_command()
    for op in operand_list:
        if not bhary.check(op):
            raise TypeError("All operands in `operand_list` must be Bohrium arrays")
    _bh.flush()
    ret_msg = _bh.user_kernel(kernel_source, operand_list, compiler_command, tag)
    if len(ret_msg) > 0:
        raise RuntimeError(ret_msg)


def dtype_to_c99(dtype):
    if np.issubdtype(dtype, np.integer):
        return "%s_t" % str(dtype)
    elif _util.dtype_equal(dtype, np.float32):
        return "float"
    elif _util.dtype_equal(dtype, np.float64):
        return "double"
    elif _util.dtype_equal(dtype, np.complex64):
        return "float complex"
    elif _util.dtype_equal(dtype, np.complex128):
        return "double complex"
    raise TypeError("dtype '%s' unsupported" % str(dtype))


def gen_function_prototype(operand_list, operand_name_list=None):
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
