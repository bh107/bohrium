import util

class test_nan:
    def init(self):
        for dtype in util.TYPES.FLOAT + util.TYPES.COMPLEX:
            yield "a = M.arange(10, dtype=%s); a[1] = M.nan; " % dtype

    def test_add_scalar(self, cmd):
        return cmd + "res = a + 42"

    def test_add_array(self, cmd):
        return cmd + "res = a + M.ones_like(a)"

