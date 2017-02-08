import util
import bohrium as bh
import bohrium.blas

def has_ext():
    try:
        a = bh.arange(4).astype(bh.float64).reshape(2, 2)
        bh.blas.gemm(a, a)
        return True
    except Exception as e:
        print("\n[ext] Cannot test BLAS extension methods.")
        print(e)
        return False


class test_ext_blas_identical:
    def init(self):
        if not has_ext():
            return

        for t in util.TYPES.FLOAT + util.TYPES.COMPLEX:
            for r in range(1, 20):
                cmd  = "a = M.arange(%d, dtype=%s).reshape(%s); " % (r*r, t, (r, r))
                cmd += "b = M.arange(%d, dtype=%s).reshape(%s); " % (r*r, t, (r, r))
                yield cmd, t

    def test_gemm(self, args):
        cmd, _ = args
        cmd_np = cmd + "res = np.dot(a, b);"
        cmd_bh = cmd + "res = bh.blas.gemm(a, b);"
        return cmd_np, cmd_bh
