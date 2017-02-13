import util

class test_sign:
    def init(self):
        for dtype in util.TYPES.ALL_SIGNED:
            yield "a = M.arange(-10, 0, 1, dtype=%s); " % dtype

    def test_sign(self, cmd):
        return cmd + "res = M.sign(a)"


class test_csign:
    def init(self):
        for dtype in util.TYPES.ALL_SIGNED:
            yield "a = M.arange(10, 0, 1, dtype=%s); " % dtype

    def test_sign(self, cmd):
        return cmd + "res = M.sign(a)"


class test_csign_neg:
    def init(self):
        for dtype in util.TYPES.ALL_SIGNED:
            yield "a = M.arange(-10, 0, 1, dtype=%s);" % dtype

    def test_sign(self, cmd):
        return cmd + "res = M.sign(a)"


class test_csign_zero:
    def init(self):
        for dtype in util.TYPES.ALL_SIGNED:
            yield "a = M.zeros(10, dtype=%s);" % dtype

    def test_sign(self, cmd):
        return cmd + "res = M.sign(a)"
