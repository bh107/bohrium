import bohrium
import util
import functools
import operator

class test_set_bool_mask:
    def init(self):
        for dtype in bohrium._info.numpy_types:
            dtype = "np.%s"%dtype.name
            for cmd, shape in util.gen_random_arrays("R", 2, min_ndim=1, samples_in_each_ndim=1,
                                                     dtype=dtype, no_views=True):
                cmd = "R = bh.random.RandomState(42); res=%s; " \
                      "m = R.random_integers(0, 1, size=res.shape, dtype=np.bool, bohrium=BH); " % cmd
                yield (cmd, dtype)

    def test_set_scalar(self, arg):
        (cmd, dtype) = arg
        cmd += "res[m] = %s(42)" % dtype
        return cmd


class test_get_bool_mask:
    def init(self):
        for ary, shape in util.gen_random_arrays("R", 3, max_dim=50, dtype="np.float64"):
            nelem = functools.reduce(operator.mul, shape)
            if nelem == 0:
                continue
            cmd = "R = bh.random.RandomState(42); a=%s; " % ary
            cmd += "m = R.random_integers(0, 1, size=a.shape, dtype=np.bool, bohrium=BH); "
            yield (cmd)

    def test_get(self, cmd):
        cmd += "res = a[m]"
        return cmd


class test_where:
    def init(self):
        for dtype in bohrium._info.numpy_types:
            dtype = "np.%s" % dtype.name
            for cmd, shape in util.gen_random_arrays("R", 2, min_ndim=1, samples_in_each_ndim=1,
                                                     dtype=dtype, no_views=True):
                cmd = "R = bh.random.RandomState(42); a = %s; " % cmd
                cmd += "b = R.random(%s, dtype=%s, bohrium=BH); " % (shape, dtype)
                cmd += "m = R.random_integers(0, 1, size=a.shape, dtype=np.bool, bohrium=BH); "
                yield (cmd, dtype)

    def test_scalar1(self, arg):
        (cmd, dtype) = arg
        cmd += "res = M.where(m, %s(42), a)" % dtype
        return cmd

    def test_scalar2(self, arg):
        (cmd, dtype) = arg
        cmd += "res = M.where(m, a, %s(42))" % dtype
        return cmd

    def test_array(self, arg):
        (cmd, dtype) = arg
        cmd += "res = M.where(m, a, b)"
        return cmd

