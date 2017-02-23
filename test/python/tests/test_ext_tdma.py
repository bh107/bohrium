import util
import bohrium as bh
import bohrium.linalg

def has_ext():
    try:
        a = bh.arange(4).astype(bh.float64).reshape(2, 2)
        bh.linalg.solve_tridiagonal(a, a, a, a)
        return True
    except Exception as e:
        print("\n\033[31m[ext] Cannot test TDMA extension methods.\033[0m")
        print(e)
        return False


class test_ext_tdma:
    def init(self):
        if not has_ext():
            return

        for t in util.TYPES.FLOAT:
            for r in (2, 10, 100):
                cmd  = "np.random.seed(123456); "
                # matrix has to be diagonally dominant for Thomas algorithm to be stable
                cmd += "a = 1 - 2 * M.array(np.random.rand(*%s), dtype=%s); " % ((2, r), t)
                cmd += "b = 100 * M.array(np.random.rand(*%s), dtype=%s); " % ((2, r), t)
                cmd += "c = 1 - 2 * M.array(np.random.rand(*%s), dtype=%s); " % ((2, r), t)
                cmd += "d = M.array(np.random.rand(*%s), dtype=%s); " % ((2, r), t)
                cmd += "matrix = M.empty(%s, dtype=%s); " % ((2, r, r), t)
                cmd += "matrix[0] = M.diag(a[0,1:], k=-1) + M.diag(b[0]) + M.diag(c[0,:-1], k=1); "
                cmd += "matrix[1] = M.diag(a[1,1:], k=-1) + M.diag(b[1]) + M.diag(c[1,:-1], k=1); "
                yield cmd, t

    def test_1d(self, args):
        cmd, _ = args
        cmd_np = cmd + "res = np.linalg.solve(matrix[0], d[0]); "
        cmd_bh = cmd + "res = bh.linalg.solve_tridiagonal(a[0], b[0], c[0], d[0]); "
        return cmd_np, cmd_bh

    def test_multidim(self, args):
        cmd, _ = args
        cmd_np = cmd + "res = np.array([np.linalg.solve(mm, dd) for mm,dd in zip(matrix,d)]); "
        cmd_bh = cmd + "res = bh.linalg.solve_tridiagonal(a, b, c, d); "
        return cmd_np, cmd_bh
