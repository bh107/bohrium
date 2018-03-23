import util
import bohrium as bh
import bohrium.lapack

import platform
from os import environ

def has_ext():
    try:
        a = bh.array([[76,25,11], [27,89,51], [18, 60, 32]], dtype=bh.float32)
        b = bh.array([[10], [7], [43]], dtype=bh.float32)
        bh.lapack.gesv(a, b)
        return True
    except Exception as e:
        print("\n\033[31m[ext] Cannot test LAPACK extension methods.\033[0m")
        print(e)
        return False


class test_ext_lapack_le:
    def init(self):
        if not has_ext():
            return

        for t in util.TYPES.FLOAT:
            yield t


    def test_gesv(self, t):
        cmd_np = "res = np.array([[-0.66108221], [9.45612621], [-16.01462746]]);"

        cmd_bh  = "a = bh.array([[76., 25., 11.], [27., 89., 51.], [18., 60., 32.]], dtype=%s);" % t
        cmd_bh += "b = bh.array([[10.], [7.], [43.]], dtype=%s);" % t
        cmd_bh += "res = bh.lapack.gesv(a, b);"
        return cmd_np, cmd_bh


    def test_gbsv(self, t):
        cmd_np = "res = np.array([-2., 3., 1., -4.]);"

        cmd_bh  = "a = bh.array([[-0.23, 2.54, -3.66, 0], [-6.98, 2.46, -2.73, -2.13], [0, 2.56, 2.46, 4.07], [0, 0, -4.78, -3.82]], dtype=%s);" % t
        cmd_bh += "b = bh.array([4.42, 27.13, -6.14, 10.50], dtype=%s).T;" % t
        cmd_bh += "res = bh.lapack.gbsv(a, b);"
        return cmd_np, cmd_bh


    def test_gtsv(self, t):
        cmd_np = "res = np.array([-4., 7., 3., -4., -3.], dtype=%s);" % t

        cmd_bh  = "a = bh.array([[3.0, 2.1, 0.0, 0.0, 0.0], [3.4, 2.3, -1.0, 0.0, 0.0], [0.0, 3.6, -5.0, 1.9, 0.0], [0.0, 0.0, 7.0, -0.9, 8.0], [0.0, 0.0, 0.0, -6.0, 7.1]], dtype=%s);" % t
        cmd_bh += "b = bh.array([2.7, -0.5, 2.6, 0.6, 2.7], dtype=%s).T;" % t
        cmd_bh += "res = bh.lapack.gtsv(a, b);"
        return cmd_np, cmd_bh

class test_ext_lapack_p:
    def init(self):
        if not has_ext():
            return

        for t in util.TYPES.FLOAT:
            cmd_np = "res = np.array([3., 2., 1.], dtype=%s);" % t
            cmd_bh  = "a = bh.array([[2., -1., 0.], [-1., 2., -1.], [0., -1., 2.]], dtype=%s);" % t
            cmd_bh += "b = bh.array([4., 0., 0.], dtype=%s).T;" % t

            yield (cmd_np, cmd_bh)

    def test_posv(self, args):
        (cmd_np, cmd_bh) = args
        cmd_bh += "res = bh.lapack.posv(a, b);"
        return cmd_np, cmd_bh


    def test_ppsv(self, args):
        (cmd_np, cmd_bh) = args
        cmd_bh += "res = bh.lapack.ppsv(a, b);"
        return cmd_np, cmd_bh


    def test_spsv(self, args):
        (cmd_np, cmd_bh) = args
        cmd_bh += "res = bh.lapack.spsv(a, b);"
        return cmd_np, cmd_bh


try:
    import scipy
    class test_ext_scipy:
        def init(self):
            if not has_ext():
                return

            for t in util.TYPES.FLOAT:
                for i in range(4, 10):
                    cmd  = "R = bh.random.RandomState(42);"
                    cmd += "a = R.random(%d, dtype=%s, bohrium=BH).reshape(%d, %d);" % (i*i, t, i, i)
                    cmd += "rhs = R.random(%d, dtype=%s, bohrium=BH);" % (i, t)
                    yield cmd

        def test_scipy(self, cmd):
            cmd_np = "import scipy.linalg.lapack;" + cmd + "res = scipy.linalg.lapack.dgesv(a[1:, 1:], rhs[1:])[2];"
            cmd_bh = cmd + "a = bh.array(a); rhs = bh.array(rhs); res = bh.lapack.gesv(a[1:, 1:], rhs[1:]);"

            return (cmd_np, cmd_bh)
except Exception as e:
    print("\n\033[31m[ext] Cannot test LAPACK extension methods against scipy.\033[0m")
    print(e)
