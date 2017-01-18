import util

class test_collect:
    def init(self):
        dtype = 'np.float32'
        cmd = "a = M.arange(%d, dtype=%s); " % (100, dtype)
        yield cmd
    #    for t in util.TYPES.SIGNED_INT:
    #        yield t

    def test_contract(self, cmd):
        cmd += "res = a / 180.0 * 3.14"
        return cmd

    def test_contract_reverse(self, cmd):
        cmd += "res = a * 3.14 / 180.0"
        return cmd
