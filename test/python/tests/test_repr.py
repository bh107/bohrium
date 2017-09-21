import util

class test_single_repr:
    def init(self):
        for t in util.TYPES.NORMAL:
            cmd = "a = M.array([1.234]).astype(%s).sum();" % t
            yield cmd

    def test_repr(self, cmd):
        cmd += "res = a.__repr__();"
        return cmd

    def test_str(self, cmd):
        cmd += "res = a.__str__();"
        cmd += "res = float(res)"
        return cmd


class test_array_repr:
    def init(self):
        for t in util.TYPES.NORMAL:
            cmd = "a = M.arange(10).astype(%s);" % t
            yield cmd

    def test_repr(self, cmd):
        cmd += "res = a.__repr__();"
        return cmd

    def test_str(self, cmd):
        cmd += "res = a.__str__();"
        return cmd
