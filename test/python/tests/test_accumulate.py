import util

class test_views:
    """ Test accumulate of all kind of views"""
    def init(self):
        for cmd, shape in util.gen_random_arrays("R", 4, dtype="np.float32"):
            cmd = "R = bh.random.RandomState(42); a = %s; " % cmd
            for i in range(len(shape)):
                yield (cmd, i)
            for i in range(len(shape)):
                yield (cmd, -i)

    def test_accumulate(self, arg):
        (cmd, axis) = arg
        cmd += "res = M.add.accumulate(a, axis=%d)" % axis
        return cmd

class test_sum:
    """ Test reduction of sum(), prod(), any(), and all()"""
    def init(self):
        for cmd, shape in util.gen_random_arrays("R", 3, dtype="np.float32"):
            cmd = "R = bh.random.RandomState(42); a = %s; " % cmd
            for op in ["cumsum", "cumprod"]:
                for axis in range(len(shape)):
                    yield (cmd, op, axis)

    def test_func(self, arg):
        (cmd, op, axis) = arg
        cmd += "res = M.%s(a, axis=%d)" % (op, axis)
        return cmd

    def test_method(self, arg):
        (cmd, op, axis) = arg
        cmd += "res = a.%s(axis=%d)" % (op, axis)
        return cmd

class test_primitives:
    def init(self):
        for op in ["add", "multiply"]:
            yield (op, "np.float64")
            yield (op, "np.bool")

    def test_vector(self, arg):
        (op, dtype) = arg
        cmd = "R = bh.random.RandomState(42); a = R.random(10, dtype=%s, bohrium=BH); " % dtype
        cmd += "res = M.%s.accumulate(a)" % op
        return cmd

class test_overwrite:
    def init(self):
        yield None

    def test_vector(self, _):
        cmd = """\
a = M.arange(10)
b = a.copy()
M.add.accumulate(a, out=b)
res = b.copy()
del b
"""
        return cmd
