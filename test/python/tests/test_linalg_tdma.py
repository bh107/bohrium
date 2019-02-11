import util
import bohrium as bh
import bohrium.linalg
import bohrium_api


class test_linalg_tdma:
    def init(self):
        if bohrium_api.stack_info.is_proxy_in_stack():
            return

        if bohrium_api.stack_info.is_opencl_in_stack():
            backend = "opencl"
        elif bohrium_api.stack_info.is_cuda_in_stack():
            backend = "cuda"
        else:
            backend = "openmp"

        for t in util.TYPES.FLOAT:
            for r in (2, 10, 100):
                cmd  = "np.random.seed(123456); "
                # matrix has to be diagonally dominant for Thomas algorithm to be stable
                testshape = (2, 2, r)
                cmd += "a = 1 - 2 * M.array(np.random.rand(*%s), dtype=%s); " % (testshape, t)
                cmd += "b = 100 * M.array(np.random.rand(*%s), dtype=%s); " % (testshape, t)
                cmd += "c = 1 - 2 * M.array(np.random.rand(*%s), dtype=%s); " % (testshape, t)
                cmd += "d = M.array(np.random.rand(*%s), dtype=%s); " % (testshape, t)
                cmd += "matrix = M.empty(%s, dtype=%s); " % (testshape + (r,), t)
                cmd += "matrix[0, 0] = M.diag(a[0,0,1:], k=-1) + M.diag(b[0,0]) + M.diag(c[0,0,:-1], k=1); "
                cmd += "matrix[0, 1] = M.diag(a[0,1,1:], k=-1) + M.diag(b[0,1]) + M.diag(c[0,1,:-1], k=1); "
                cmd += "matrix[1, 0] = M.diag(a[1,0,1:], k=-1) + M.diag(b[1,0]) + M.diag(c[1,0,:-1], k=1); "
                cmd += "matrix[1, 1] = M.diag(a[1,1,1:], k=-1) + M.diag(b[1,1]) + M.diag(c[1,1,:-1], k=1); "
                yield cmd, t, backend

    def test_1d(self, args):
        cmd, _, backend = args
        cmd_np = cmd + "res = np.linalg.solve(matrix[0, 0], d[0, 0]); "
        cmd_bh = (
            cmd + "res = bh.linalg.solve_tridiagonal(a[0, 0], b[0, 0], c[0, 0], d[0, 0], backend='%s'); "
            % backend
        )
        return cmd_np, cmd_bh

    def test_2d(self, args):
        cmd, _, backend = args
        cmd_np = (
            cmd +
            "res = np.array([np.linalg.solve(mm, dd) for mm,dd in zip(matrix[0],d[0])]); "
        )
        cmd_bh = (
            cmd +
            "res = bh.linalg.solve_tridiagonal(a[0], b[0], c[0], d[0], backend='%s'); "
            % backend
        )
        return cmd_np, cmd_bh

    def test_3d(self, args):
        cmd, _, backend = args
        cmd_np = (
            cmd + 
            "flatshp = (matrix.shape[0] * matrix.shape[1], matrix.shape[2], matrix.shape[3]); " +
            "res = np.array([np.linalg.solve(mm, dd) for mm,dd in zip(matrix.reshape(flatshp),d.reshape(flatshp[:-1]))]); " +
            "res = res.reshape(matrix.shape[:-1]); "
        )
        cmd_bh = cmd + "res = bh.linalg.solve_tridiagonal(a, b, c, d, backend='%s'); " % backend
        return cmd_np, cmd_bh
