import util

class test_empty:
    def init(self):
        cmd = "a = M.array([]); a = M.array([]); "
        yield cmd

    def test_add(self, cmd):
        cmd += "res = a + a"
        return cmd
