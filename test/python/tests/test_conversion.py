import util
import sys

class test_conversion:
    def init(self):
        yield "a = M.arange(1);"

    def test_float(self, cmd):
        cmd += "res = float(a);"
        return cmd

    def test_int(self, cmd):
        cmd += "res = int(a);"
        return cmd

    if sys.version_info[0] < 3:
        def test_oct(self, cmd):
            cmd += "res = oct(a);"
            return cmd

        def test_hex(self, cmd):
            cmd += "res = hex(a);"
            return cmd

        def test_long(self, cmd):
            cmd += "res = long(a);"
            return cmd
