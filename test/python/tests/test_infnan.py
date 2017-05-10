import util

class test_nan:
    def init(self):
        for dtype in util.TYPES.FLOAT + util.TYPES.COMPLEX:
            yield "a = M.arange(10, dtype=%s); a[1] = M.nan; " % dtype

    def test_add_scalar(self, cmd):
        return cmd + "res = a + 42"

    def test_add_array(self, cmd):
        return cmd + "res = a + M.ones_like(a)"


class test_inf:
    def init(self):
        for dtype in util.TYPES.FLOAT + util.TYPES.COMPLEX:
            yield "a = M.arange(10, dtype=%s); a[1] = M.inf; a[2] = -M.inf; " % dtype

    def test_add_scalar(self, cmd):
        return cmd + "res = a + 42"

    def test_add_scalar_inf(self, cmd):
        return cmd + "res = a + M.inf"

    def test_add_array(self, cmd):
        return cmd + "res = a + M.ones_like(a)"

    def test_isnan(self, cmd):
        return cmd + "res = M.isnan(a)"

    def test_isinf(self, cmd):
        return cmd + "res = M.isinf(a)"

    def test_isfinte(self, cmd):
        return cmd + "res = M.isfinite(a)"


class test_isfinite:
    def init(self):
        for dtype in util.TYPES.ALL:
            yield "a = M.arange(10, dtype=%s);" % dtype

    def test_isnan(self, cmd):
        return cmd + "res = M.isnan(a)"

    def test_isinf(self, cmd):
        return cmd + "res = M.isinf(a)"

    def test_isfinte(self, cmd):
        return cmd + "res = M.isfinite(a)"
