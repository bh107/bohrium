import util
import bohrium.blas

class test_ext_blas:
    def init(self):
        for t in util.TYPES.FLOAT + util.TYPES.COMPLEX:
            rows = 5
            cols = 5
            for r in range(1, rows):
                for c in range(1, cols):
                    cmd  = "a = M.arange(%d, dtype=%s).reshape(%s); " % (r*c, t, (r, c))
                    cmd += "b = M.arange(%d, dtype=%s).reshape(%s); " % (c*r, t, (c, r))
                    yield cmd

    def test_gemm(self, cmd):
        # Tests that 'bh.blas.gemm(a, b)' calculates the same as 'np.dot(a, b)'
        cmd_np = cmd + "res = np.dot(a, b);"
        cmd_bh = cmd + "res = bh.blas.gemm(a, b);"
        return cmd_np, cmd_bh
