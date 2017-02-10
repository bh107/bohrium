import bohrium
import util

class test_bh_opcodes:
    def init(self):
        for op in bohrium._info.op.values():
            if op['name'] not in ["identity", "sign"]:
                for type_sig in op['type_sig']:
                    yield (op, type_sig)

    def test_ufunc(self, arg):
        (op, type_sig) = arg

        cmd = "R = bh.random.RandomState(42); "

        for i, dtype in enumerate(type_sig[1:]):
            cmd += "a%d = R.random(10, dtype=np.%s, bohrium=BH); " % (i, dtype)

        cmd += "res = M.%s(" % (op['name'])

        for i in range(op['nop']-1):
            cmd += "a%d, " % i

        cmd = cmd[:-2] + ");"
        return cmd


class test_extra_binary_ops:
    def init(self):
        for op in ["true_divide", "floor_divide"]:
            for dtype in ["float64", "int64", "uint64"]:
                yield (op, dtype)

    def test_ufunc(self, arg):
        (op, dtype) = arg

        cmd =  "R = bh.random.RandomState(42); "
        cmd += "a0 = R.random(10, dtype=np.%s, bohrium=BH); "%dtype
        cmd += "a1 = R.random(10, dtype=np.%s, bohrium=BH); "%dtype
        cmd += "res = M.%s(a0, a1)"%op

        return cmd
