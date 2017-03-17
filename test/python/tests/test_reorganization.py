import util
import functools
import operator


class test_gather:
    def init(self):
        for cmd, shape in util.gen_random_arrays("R", 3, min_ndim=1, dtype="np.uint32"):
            if functools.reduce(operator.mul, shape) > 0:
                cmd = "R = bh.random.RandomState(42); a = %s; " % cmd
                yield cmd

    def test_take(self, cmd):
        return cmd + "res = M.take(a, a % a.shape[0])"
