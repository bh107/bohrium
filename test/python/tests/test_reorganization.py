import util
import functools
import operator


class test_general:
    def init(self):
        for ary, shape in util.gen_random_arrays("R", 3, max_dim=50, dtype="np.uint32"):
            nelem = functools.reduce(operator.mul, shape)
            if nelem == 0:
                continue
            cmd = "R = bh.random.RandomState(42); a = %s; " % ary
            cmd += "ind = M.arange(%d).reshape(%s); " % (nelem, shape)
            yield cmd
            yield cmd + "ind = ind[::2]; "
            if shape[0] > 2:
                yield cmd + "ind = ind[1:]; "
            if len(shape) > 1 and shape[1] > 5:
                yield cmd + "ind = a[3:]; "

    def test_take(self, cmd):
        return cmd + "res = M.take(a, ind)"

    def test_take_ary_mth(self, cmd):
        return cmd + "res = a.take(ind)"

    def test_indexing(self, cmd):
        return cmd + "res = a.flatten()[ind.flatten()]"


class test_scatter:
    def init(self):
        for cmd, shape in util.gen_random_arrays("R", 3, min_ndim=1, dtype="np.uint32"):
            if functools.reduce(operator.mul, shape) > 0:
                cmd = "R = bh.random.RandomState(42); res = %s; " % cmd
                yield cmd

    def test_put(self, cmd):
        return cmd + "M.put(res, res % res.shape[0], res)"
