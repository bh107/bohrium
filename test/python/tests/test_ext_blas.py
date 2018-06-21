import util
import bohrium as bh
import bohrium.blas

import platform
from os import environ
os = platform.system()
if os == "Darwin" and environ.get("BH_STACK", "") == "opencl":
    print("\033[31m[EXT] Ignoring 64-bit OpenCL clBLAS tests on MacOS.\n[EXT] Intel graphics does not support 64-bit float and complex types.\033[0m")
    float_types   = ['np.float32']
    complex_types = []
else:
    float_types   = util.TYPES.FLOAT
    complex_types = util.TYPES.COMPLEX

def has_ext():
    try:
        a = bh.arange(4).astype(bh.float64).reshape(2, 2)
        bh.blas.gemm(a, a)
        return True
    except Exception as e:
        print("\n\033[31m[ext] Cannot test BLAS extension methods.\033[0m")
        print(e)
        return False


class test_ext_blas_identical:
    def init(self):
        if not has_ext():
            return

        for t in float_types + complex_types:
            for r in range(1, 20):
                cmd  = "a = M.arange(%d, dtype=%s).reshape(%s); " % (r*r, t, (r, r))
                cmd += "b = M.arange(%d, dtype=%s).reshape(%s); " % (r*r, t, (r, r))
                yield cmd, t

    def test_gemm(self, args):
        cmd, _ = args
        cmd_np = cmd + "res = np.dot(a, b);"
        cmd_bh = cmd + "res = bh.blas.gemm(a, b);"
        return cmd_np, cmd_bh

    def test_gemmt(self, args):
        cmd, t = args
        if t in util.TYPES.COMPLEX:
            return "res = 0"

        cmd_np = cmd + "res = np.dot(a.T, b);"
        cmd_bh = cmd + "res = bh.blas.gemmt(a, b);"
        return cmd_np, cmd_bh

    def test_syr2k(self, args):
        cmd, _ = args
        cmd_np = cmd + "res = np.triu(np.dot(a, b.transpose()) + np.dot(b, a.transpose()));"
        cmd_bh = cmd + "res = bh.blas.syr2k(a, b);"
        return cmd_np, cmd_bh

    def test_her2k(self, args):
        cmd, t = args

        if t not in complex_types:
            return "res = 0;"

        cmd_np = cmd + "res = np.triu(np.dot(a, b.transpose()) + np.dot(b, a.transpose()));"
        cmd_bh = cmd + "res = bh.blas.her2k(a, b);"
        return cmd_np, cmd_bh


class test_ext_blas_symmetric:
    def init(self):
        if not has_ext():
            return

        for t in float_types + complex_types:
            for r in range(2, 10):
                # a will be symmetric/hermitian
                cmd  = "a = M.arange(%d, dtype=%s).reshape(%s); a = ((a + a.T)/2);" % (r*r, t, (r, r))
                cmd += "b = M.arange(%d, dtype=%s).reshape(%s);" % (r*r, t, (r, r))
                yield cmd, t

    def test_symm(self, args):
        cmd, _ = args
        cmd_np = cmd + "res = np.dot(a, b);"
        cmd_bh = cmd + "res = bh.blas.symm(a, b);"
        return cmd_np, cmd_bh

    def test_hemm(self, args):
        cmd, t = args

        if t not in complex_types:
            return "res = 0;"

        cmd_np = cmd + "res = np.dot(a, b);"
        cmd_bh = cmd + "res = bh.blas.hemm(a, b);"
        return cmd_np, cmd_bh


class test_ext_blas_only_a:
    def init(self):
        if not has_ext():
            return

        for t in complex_types:
            for r in range(2, 10):
                cmd  = "a = M.arange(%d, dtype=%s).reshape(%s);" % (r*r, t, (r, r))
                yield cmd, t

    def test_syrk(self, args):
        cmd, _ = args
        cmd_np = cmd + "res = np.triu(np.dot(a, a.transpose()));"
        cmd_bh = cmd + "res = bh.zeros_like(a); bh.blas.syrk(a, c=res);"
        return cmd_np, cmd_bh

    def test_herk(self, args):
        cmd, t = args

        if t not in complex_types:
            return "res = 0;"

        cmd_np = cmd + "res = np.triu(np.dot(a, a.transpose()));"
        cmd_bh = cmd + "res = bh.zeros_like(a); bh.blas.herk(a, c=res);"
        return cmd_np, cmd_bh


class test_ext_blas_unit_triangular:
    def init(self):
        if not has_ext():
            return

        for t in float_types + complex_types:
            for r in range(2, 10):
                cmd  = "a = np.arange(%d, dtype=%s).reshape(%s); np.fill_diagonal(a, 1); a = np.triu(a);" % (r*r, t, (r, r))
                cmd += "b = M.arange(%d, dtype=%s).reshape(%s);" % (r*r, t, (r, r))
                yield cmd, t

    def test_trmm(self, args):
        cmd, _ = args
        cmd_np = cmd + "res = np.dot(a, b);"
        cmd_bh = cmd + "a = bh.array(a); res = bh.blas.trmm(a, b);"
        return cmd_np, cmd_bh

    def test_trsm(self, args):
        cmd, _ = args
        cmd_np = cmd + "from numpy.linalg import inv; res = np.dot(inv(a), b);"
        cmd_bh = cmd + "a = bh.array(a); res = bh.blas.trsm(a, b);"
        return cmd_np, cmd_bh


class test_ext_cg:
    def init(self):
        if not has_ext():
            return

        for t in float_types:
            yield t

    def test_cg(self, t):
        cmd_np = "res = np.array([0.09090909, 0.63636364], dtype=%s)" % t
        cmd_bh  = "a = bh.array([[4, 1], [1, 3]], dtype=%s);" % t
        cmd_bh += "b = bh.array([1, 2], dtype=%s);" % t
        cmd_bh += "x = bh.array([2, 1], dtype=%s);" % t
        cmd_bh += "res = bh.linalg.cg(a, b, x);"
        return cmd_np, cmd_bh
