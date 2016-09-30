import util


class tes1t_reduce_views:
    """ Test reduction of all kind of views"""
    def init(self):
        for cmd, ndim in util.gen_random_arrays("R", 4, dtype="np.float32"):
            cmd = "R = bh.random.RandomState(42); a = %s; " % cmd
            for i in range(ndim):
                yield (cmd, i)
            for i in range(ndim):
                yield (cmd, -i)

    def test_reduce(self, (cmd, axis)):
        cmd += "res = M.add.reduce(a, axis=%d)" % axis
        return cmd


class tes1t_reduce_sum:
    """ Test reduction of sum() and prod()"""
    def init(self):
        for cmd, ndim in util.gen_random_arrays("R", 3, dtype="np.float32"):
            cmd = "R = bh.random.RandomState(42); a = %s; " % cmd
            for op in ["sum", "prod"]:
                yield (cmd, op)

    def test_func(self, (cmd, op)):
        cmd += "res = M.%s(a)" % op
        return cmd

    def test_method(self, (cmd, op)):
        cmd += "res = a.%s()" % op
        return cmd


class test_reduce_primitives:
    def init(self):
        for op in ["add", "multiply", "minimum", "maximum"]:
            yield (op, "np.float64")

        for op in ["bitwise_or", "bitwise_xor"]:
            yield (op, "np.uint64")

        for op in ["add", "logical_or", "logical_and", "logical_xor"]:
            yield (op, "np.bool")

    def test_vector(self, (op, dtype)):
        cmd = "R = bh.random.RandomState(42); a = R.random(10, dtype=%s, bohrium=BH); " % dtype
        cmd += "res = M.%s.reduce(a)" % op
        return cmd

