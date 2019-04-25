import util


def bh107_test(func):
    def inner(self, args):
        cmd = func(self, args)
        return (cmd, "import bh107 as bh; %s; res = res.toNumPy()" % cmd)

    return inner


class test_array_create:
    def init(self):
        for t in util.TYPES.ALL:
            yield t

    @bh107_test
    def test_zeros(self, dtype):
        return "res = M.zeros(%d, dtype=%s)" % (100, dtype)

    @bh107_test
    def test_ones(self, dtype):
        return "res = M.ones(%d, dtype=%s)" % (100, dtype)
