import bohrium
import numpy


class test_bh_opcodes:
    def init(self):
        for op in bohrium._info.op.values():
            if op["name"] not in ["identity", "sign"] and op['elementwise']:
                for type_sig in op["type_sig"]:
                    yield (op, type_sig)

    def test_ufunc(self, arg):
        (op, type_sig) = arg

        cmd = "R = bh.random.RandomState(42); "

        for i, dtype in enumerate(type_sig[1:]):
            cmd += "a%d = R.random(10, dtype=np.%s, bohrium=BH); " % (i, dtype)

        if op["name"] == "arccosh":
            cmd += "a%d += 1;" % i

        cmd += "res = M.%s(" % (op["name"])

        for i in range(op["nop"]-1):
            cmd += "a%d, " % i

        cmd = cmd[:-2] + ");"
        return cmd


class test_bh_operators:
    def init(self):
        for op in ['+', '-', '*', '/', '//', '%', '==']:
            for dtype in ['float64', 'int64']:
                yield (op, dtype)

    def test_arrays(self, arg):
        (op, dtype) = arg
        cmd = "R = bh.random.RandomState(42); "
        cmd += "a1 = R.random(10, dtype=np.%s, bohrium=BH); " % dtype
        cmd += "a2 = R.random(10, dtype=np.%s, bohrium=BH) + 1; " % dtype
        cmd += "res = a1 %s a2" % op
        return cmd

    def test_scalar_rhs(self, arg):
        (op, dtype) = arg
        cmd = "R = bh.random.RandomState(42); "
        cmd += "a1 = R.random(10, dtype=np.%s, bohrium=BH); " % dtype
        cmd += "a2 = np.%s(42); " % dtype
        cmd += "res = a1 %s a2" % op
        return cmd


class test_bh_operators_lhs:
    def init(self):
        if numpy.__version__ >= "1.13":
            for op in ['+', '-', '*', '/', '//', '%', '==']:
                for dtype in ['float64', 'int64']:
                    yield (op, dtype)
        else:
            print("The version of NumPy is too old (<= 1.13), ignoring test")

    def test_scalar_lhs(self, arg):
        (op, dtype) = arg
        cmd = "R = bh.random.RandomState(42); "
        cmd += "a1 = np.%s(42); " % dtype
        cmd += "a2 = R.random(10, dtype=np.%s, bohrium=BH) + 1; " % dtype
        cmd += "res = a1 %s a2" % op
        return cmd


class test_extra_binary_ops:
    def init(self):
        for op in ["true_divide", "floor_divide"]:
            for dtype in ["float64", "int64", "uint64"]:
                yield (op, dtype)

    def test_ufunc(self, arg):
        (op, dtype) = arg

        cmd =  "R = bh.random.RandomState(42); "
        cmd += "a0 = R.random(10, dtype=np.%s, bohrium=BH); " % dtype
        cmd += "a1 = R.random(10, dtype=np.%s, bohrium=BH); " % dtype
        cmd += "res = M.%s(a0, a1)" % op
        return cmd


class test_power:
    def init(self):
        for op in ["power"]:
            for dtype in ["float32", "float64"]:
                yield (op, dtype)

    def test_ufunc(self, arg):
        (op, dtype) = arg

        cmd =  "R = bh.random.RandomState(42); "
        cmd += "a0 = R.random(10, dtype=np.%s, bohrium=BH); " % dtype
        cmd += "res = M.%s(a0, 1.42)" % op
        return cmd
