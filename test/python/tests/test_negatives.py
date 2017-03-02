import util


class test_negative:
    def init(self):
        for dtype in util.TYPES.SIGNED_INT:
            yield "a = M.arange(-11, -1, 1, dtype=%s); b = M.arange(1, 11, dtype=%s);" % (dtype, dtype)
            yield "b = M.arange(-11, -1, 1, dtype=%s); a = M.arange(1, 11, dtype=%s);" % (dtype, dtype)

    def test_division(self, cmd):
        return cmd + "res = a / b"

    def test_remainder(self, cmd):
        return cmd + "res = M.remainder(a, b)"

    def test_mod(self, cmd):
        return cmd + "res = M.mod(a, b)"

    def test_fmod(self, cmd):
        return cmd + "res = M.fmod(a, b)"

    def test_pct(self, cmd):
        return cmd + "res = a % b"


