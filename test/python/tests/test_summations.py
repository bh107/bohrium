import util
import functools
import operator


class test_sums:
    """ Test reduction of sum(), prod(), any(), and all()"""
    def init(self):
        for cmd, shape in util.gen_random_arrays("R", 3, dtype="np.float32"):
            cmd = "R = bh.random.RandomState(42); a = %s; " % cmd
            nelem = functools.reduce(operator.mul, shape)
            if nelem > 0:
                for op in ["sum", "prod", "all", "any", "mean"]:
                    yield (cmd, op, None)
                    for axis in range(len(shape)):
                        yield (cmd, op, axis)

    def test_func(self, arg):
        (cmd, op, axis) = arg
        if axis is None:
            cmd += "res = M.%s(a)" % op
        else:
            cmd += "res = M.%s(a, axis=%d)" % (op, axis)
        return cmd

    def test_method(self, arg):
        (cmd, op, axis) = arg
        if axis is None:
            cmd += "res = a.%s()" % op
        else:
            cmd += "res = a.%s(axis=%d)" % (op, axis)
        return cmd


class test_argminmax:
    def init(self):
        for cmd, shape in util.gen_random_arrays("R", 3, dtype="np.float32"):
            cmd = "R = bh.random.RandomState(42); a = %s; " % cmd
            nelem = functools.reduce(operator.mul, shape)
            if nelem > 0:
                for op in ['argmin', 'argmax']:
                    yield (cmd, op)

    def test_func(self, args):
        (cmd, op) = args
        return cmd + "res = M.%s(a)" % op

    def test_method(self, args):
        (cmd, op) = args
        return cmd + "res = a.%s()" % op

