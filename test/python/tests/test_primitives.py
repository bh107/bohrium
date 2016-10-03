import bohrium

class test_bh_opcodes:
    def init(self):
        for op in bohrium._info.op.values():
            if op['name'] not in ["identity", "sign"]:
                for type_sig in op['type_sig']:
                    yield (op, type_sig)

    def test_ufunc(self, (op, type_sig)):
        cmd = "R = bh.random.RandomState(42); "
        for i, dtype in enumerate(type_sig[1:]):
            cmd += "a%d = R.random(10, dtype=np.%s, bohrium=<BH>); " % (i, dtype)
        cmd += "res = M.%s(" % (op['name'])
        for i in range(op['nop']-1):
            cmd += "a%d, " % i
        cmd = cmd[:-2] + ");"
        cmd_np = cmd.replace("<BH>", "False")
        cmd_bh = cmd.replace("<BH>", "True")
        return cmd_np, cmd_bh

