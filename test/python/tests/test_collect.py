import util

class test_collect:
    def init(self):
        for t in util.TYPES.ALL:
            cmd = "a = M.arange(%d, dtype=%s); " % (100, t)
            yield cmd

    def test_contract(self, cmd):
        cmd += "res = a / 180.0 * 3.14"
        return cmd

    def test_contract_reverse(self, cmd):
        cmd += "res = a * 3.14 / 180.0"
        return cmd
